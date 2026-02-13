import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist


@dataclass
class ClusteringResult:
    metric_used: str
    linkage_matrix: np.ndarray
    metric_choice: dict | None
    distance_condensed: np.ndarray


def _pairwise_complete_distance(X, metric):
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported metric for pairwise complete obs: {metric}")

    n_samples = X.shape[0]
    out = np.empty(n_samples * (n_samples - 1) // 2, dtype=np.float64)
    k = 0
    for i in range(n_samples - 1):
        xi = X[i]
        for j in range(i + 1, n_samples):
            xj = X[j]
            valid = np.isfinite(xi) & np.isfinite(xj)
            if not np.any(valid):
                raise ValueError(
                    "Pairwise complete observation mode found a pair with no overlapping valid features."
                )
            ai = xi[valid]
            bj = xj[valid]
            if metric == "euclidean":
                diff = ai - bj
                out[k] = np.sqrt(np.dot(diff, diff))
            else:
                norm = np.linalg.norm(ai) * np.linalg.norm(bj)
                if norm == 0.0:
                    out[k] = 1.0
                else:
                    out[k] = 1.0 - (np.dot(ai, bj) / norm)
            k += 1
    return out


def compute_condensed_distance(X, metric, pairwise_complete_obs=True):
    X = np.asarray(X, dtype=np.float64)
    if not pairwise_complete_obs:
        return pdist(X, metric=metric)

    # Fast path: if there are no missing values, pairwise-complete equals regular pdist.
    if np.isfinite(X).all():
        return pdist(X, metric=metric)

    return _pairwise_complete_distance(X, metric)


def _compute_ccc_from_distance(distance_condensed, method):
    Z = linkage(distance_condensed, method=method)
    ccc, _ = cophenet(Z, distance_condensed)
    return float(ccc), Z


def _compute_with_metric(X, method, metric, pairwise_complete_obs):
    dist = compute_condensed_distance(X, metric, pairwise_complete_obs=pairwise_complete_obs)
    ccc, Z = _compute_ccc_from_distance(dist, method)
    return ccc, Z, dist


def select_metric_and_linkage(
    X,
    linkage_method="complete",
    metric="auto",
    threshold=0.02,
    out_dir=None,
    pairwise_complete_obs=True,
):
    linkage_method = linkage_method.lower()
    metric = metric.lower()

    if linkage_method == "ward":
        if metric == "cosine":
            raise ValueError("Cosine metric is not allowed with ward linkage.")
        ccc_euclid, Z, dist = _compute_with_metric(
            X, linkage_method, "euclidean", pairwise_complete_obs=pairwise_complete_obs
        )
        metric_choice = None
        if metric == "auto":
            metric_choice = {
                "metric_requested": metric,
                "metric_selected": "euclidean",
                "ccc_euclidean": ccc_euclid,
                "ccc_cosine": None,
                "threshold": threshold,
                "pairwise_complete_obs": bool(pairwise_complete_obs),
            }
            if out_dir:
                _save_metric_choice(out_dir, metric_choice)
        return ClusteringResult("euclidean", Z, metric_choice, dist)

    if metric in ("euclidean", "cosine"):
        _ccc, Z, dist = _compute_with_metric(
            X, linkage_method, metric, pairwise_complete_obs=pairwise_complete_obs
        )
        return ClusteringResult(metric, Z, None, dist)

    if metric != "auto":
        raise ValueError("metric must be one of: euclidean, cosine, auto")

    ccc_euclid, Z_euclid, dist_euclid = _compute_with_metric(
        X, linkage_method, "euclidean", pairwise_complete_obs=pairwise_complete_obs
    )
    ccc_cosine, Z_cosine, dist_cosine = _compute_with_metric(
        X, linkage_method, "cosine", pairwise_complete_obs=pairwise_complete_obs
    )

    if (ccc_cosine - ccc_euclid) >= threshold:
        metric_selected = "cosine"
        Z = Z_cosine
        dist = dist_cosine
    else:
        metric_selected = "euclidean"
        Z = Z_euclid
        dist = dist_euclid

    metric_choice = {
        "metric_requested": "auto",
        "metric_selected": metric_selected,
        "ccc_euclidean": ccc_euclid,
        "ccc_cosine": ccc_cosine,
        "threshold": threshold,
        "pairwise_complete_obs": bool(pairwise_complete_obs),
    }

    if out_dir:
        _save_metric_choice(out_dir, metric_choice)

    return ClusteringResult(metric_selected, Z, metric_choice, dist)


def _save_metric_choice(out_dir, metric_choice):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metric_choice.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metric_choice, f, indent=2)


def get_cluster_labels(Z, n_clusters):
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels.astype(int)


def _cluster_memberships(Z):
    n_samples = Z.shape[0] + 1
    members = {i: frozenset([i]) for i in range(n_samples)}
    clusters = {}
    for merge_idx, row in enumerate(Z):
        left = int(row[0])
        right = int(row[1])
        node_id = n_samples + merge_idx
        merged = members[left].union(members[right])
        members[node_id] = merged
        if len(merged) < n_samples:
            clusters[node_id] = merged
    return clusters


def compute_bootstrap_support(
    X,
    Z_reference,
    linkage_method,
    metric,
    n_bootstrap=10000,
    seed=123,
    pairwise_complete_obs=True,
):
    if n_bootstrap is None or n_bootstrap <= 0:
        return None

    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    if n_features < 2:
        raise ValueError("Bootstrap requires at least 2 features.")

    ref_clusters = _cluster_memberships(Z_reference)
    ref_items = list(ref_clusters.items())
    counts = {node_id: 0 for node_id in ref_clusters.keys()}

    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled_cols = rng.integers(0, n_features, size=n_features, endpoint=False)
        Xb = X[:, sampled_cols]
        dist_b = compute_condensed_distance(
            Xb, metric=metric, pairwise_complete_obs=pairwise_complete_obs
        )
        Zb = linkage(dist_b, method=linkage_method)
        cluster_set = set(_cluster_memberships(Zb).values())
        for node_id, cluster_members in ref_items:
            if cluster_members in cluster_set:
                counts[node_id] += 1

    return {
        node_id: {
            "cluster_size": len(ref_clusters[node_id]),
            "support_fraction": counts[node_id] / float(n_bootstrap),
            "support_percent": (counts[node_id] * 100.0) / float(n_bootstrap),
        }
        for node_id in ref_clusters.keys()
    }


def save_bootstrap_support_csv(out_dir, bootstrap_support):
    if not bootstrap_support:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bootstrap_support.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "cluster_size", "bootstrap_support_percent"])
        for node_id in sorted(bootstrap_support.keys()):
            row = bootstrap_support[node_id]
            writer.writerow([node_id, row["cluster_size"], round(row["support_percent"], 6)])
    return path


def save_dendrogram(
    Z,
    labels,
    out_path,
    cluster_labels=None,
    leaf_spacing=0.12,
    max_width=140.0,
):
    n = len(labels)
    width = min(max(14.0, n * float(leaf_spacing)), float(max_width))
    height = 9.0

    # Dynamic font sizing keeps labels legible across large leaf counts.
    leaf_font_size = max(4.0, min(9.0, 220.0 / max(1, n)))

    fig, ax = plt.subplots(figsize=(width, height))
    ddata = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
        ax=ax,
    )
    ax.set_ylabel("Distance")

    leaves = ddata["leaves"]
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels)
        unique = np.unique(cluster_labels)
        cmap = plt.get_cmap("tab20", len(unique))
        color_lookup = {int(cluster): cmap(i) for i, cluster in enumerate(unique)}
        leaf_colors = [color_lookup[int(cluster_labels[idx])] for idx in leaves]
    else:
        leaf_colors = ddata.get("leaves_color_list", ["#666666"] * len(leaves))

    y_max = max(1e-9, ax.get_ylim()[1])
    y_square = -0.05 * y_max
    x_positions = np.arange(5, 10 * len(leaves) + 5, 10)
    ax.scatter(
        x_positions,
        np.full(len(leaves), y_square),
        c=leaf_colors,
        marker="s",
        s=30,
        linewidths=0.3,
        edgecolors="black",
        clip_on=False,
        zorder=5,
    )
    ax.set_ylim(y_square * 1.8, y_max)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return leaves

