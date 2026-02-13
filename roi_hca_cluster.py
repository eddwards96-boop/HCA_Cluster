import argparse
import csv
import json
import os

import numpy as np

from roi_tools import select_roi_interactive, save_roi, load_roi
from features import build_feature_matrix
from clustering import (
    compute_bootstrap_support,
    get_cluster_labels,
    save_bootstrap_support_csv,
    save_dendrogram,
    select_metric_and_linkage,
)
from report import generate_report_html


def cmd_select_roi(args):
    verts, width, height = select_roi_interactive(args.image)
    if verts is None:
        raise SystemExit("ROI selection cancelled or empty.")

    try:
        roi_json, mask_path = save_roi(args.out, width, height, verts, notes=args.notes or "")
    except ValueError as e:
        raise SystemExit(str(e))

    print(f"Saved ROI json: {roi_json}")
    print(f"Saved ROI mask: {mask_path}")


def _write_leaf_order(out_dir, leaf_order, filenames):
    path = os.path.join(out_dir, "leaf_order.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order_index", "filename"])
        for order_index, idx in enumerate(leaf_order, start=1):
            writer.writerow([order_index, filenames[idx]])
    return path


def _write_clusters(out_dir, filenames, leaf_order, cluster_labels):
    order_index_map = {idx: i + 1 for i, idx in enumerate(leaf_order)}
    path = os.path.join(out_dir, "clusters.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cluster_label", "order_index"])
        for i, filename in enumerate(filenames):
            writer.writerow([filename, int(cluster_labels[i]), order_index_map[i]])
    return path


def cmd_cluster(args):
    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    if not os.path.isfile(args.roi):
        raise SystemExit(f"ROI json not found: {args.roi}")

    if args.bootstrap is not None and args.bootstrap < 0:
        raise SystemExit("--bootstrap must be >= 0")
    if args.leaf_spacing <= 0:
        raise SystemExit("--leaf_spacing must be > 0")
    if args.max_fig_width <= 0:
        raise SystemExit("--max_fig_width must be > 0")

    os.makedirs(args.out, exist_ok=True)

    _roi_data, roi_mask = load_roi(args.roi)

    try:
        X, filenames, image_paths, info, _mask_ds = build_feature_matrix(
            args.input_dir,
            roi_mask,
            downsample_factor=args.downsample_factor,
            max_pixels=args.max_pixels,
            seed=args.seed,
            pca_components=args.pca_components,
        )
    except Exception as e:
        raise SystemExit(str(e))

    try:
        clustering_result = select_metric_and_linkage(
            X,
            linkage_method=args.linkage,
            metric=args.metric,
            threshold=args.metric_threshold,
            out_dir=args.out if args.metric == "auto" else None,
            pairwise_complete_obs=args.pairwise_complete_obs,
        )
    except Exception as e:
        raise SystemExit(str(e))

    metric_used = clustering_result.metric_used
    Z = clustering_result.linkage_matrix
    metric_choice = clustering_result.metric_choice

    if metric_choice is not None:
        print("Metric choice:")
        print(json.dumps(metric_choice, indent=2))

    cluster_labels = None
    if args.n_clusters is not None:
        if args.n_clusters < 2:
            raise SystemExit("--n_clusters must be >= 2")
        cluster_labels = get_cluster_labels(Z, args.n_clusters)

    dendro_path = os.path.join(args.out, "dendrogram.png")
    leaf_order = save_dendrogram(
        Z,
        filenames,
        dendro_path,
        cluster_labels=cluster_labels,
        leaf_spacing=args.leaf_spacing,
        max_width=args.max_fig_width,
    )

    _write_leaf_order(args.out, leaf_order, filenames)

    if args.n_clusters is not None:
        _write_clusters(args.out, filenames, leaf_order, cluster_labels)

    bootstrap_support = None
    if args.bootstrap is not None and args.bootstrap > 0:
        print(f"Running bootstrap support with {args.bootstrap} replicates...")
        try:
            bootstrap_support = compute_bootstrap_support(
                X,
                Z_reference=Z,
                linkage_method=args.linkage,
                metric=metric_used,
                n_bootstrap=args.bootstrap,
                seed=args.bootstrap_seed,
                pairwise_complete_obs=args.pairwise_complete_obs,
            )
        except Exception as e:
            raise SystemExit(f"Bootstrap failed: {e}")
        save_bootstrap_support_csv(args.out, bootstrap_support)

    features_info = {
        "num_images": info.num_images,
        "roi_pixels_original": info.roi_pixels_original,
        "roi_pixels_after_downsample": info.roi_pixels_after_downsample,
        "roi_pixels_used": info.roi_pixels_used,
        "roi_sampled": info.sampled,
        "max_pixels": info.max_pixels,
        "downsample_factor": info.downsample_factor,
        "pca_components": info.pca_components,
        "final_dim": info.final_dim,
        "linkage": args.linkage,
        "metric": metric_used,
        "pairwise_complete_obs": args.pairwise_complete_obs,
        "bootstrap_replicates": args.bootstrap,
    }
    with open(os.path.join(args.out, "features_info.json"), "w", encoding="utf-8") as f:
        json.dump(features_info, f, indent=2)

    roi_report = {
        "extraction_mode": "per_pixel_rgb",
        "aggregation": "none",
        "vector_layout": "[R1,G1,B1, R2,G2,B2, ...] over ROI pixels in row-major order",
        "roi_pixels_original": info.roi_pixels_original,
        "roi_pixels_after_downsample": info.roi_pixels_after_downsample,
        "roi_pixels_used": info.roi_pixels_used,
        "downsample_factor": info.downsample_factor,
        "max_pixels": info.max_pixels,
        "sampled": info.sampled,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "roi_extraction_report.json"), "w", encoding="utf-8") as f:
        json.dump(roi_report, f, indent=2)

    if args.report_html:
        generate_report_html(
            args.out,
            dendro_path,
            image_paths,
            leaf_order,
            cluster_labels=cluster_labels,
        )

    if bootstrap_support:
        support_values = np.array([v["support_percent"] for v in bootstrap_support.values()], dtype=float)
        print(
            "Bootstrap support summary "
            f"(min/median/max %): {support_values.min():.2f} / "
            f"{np.median(support_values):.2f} / {support_values.max():.2f}"
        )
    print("ROI extraction summary:")
    print(" - mode: per-pixel RGB (no area averaging)")
    print(f" - ROI pixels (original): {info.roi_pixels_original}")
    print(f" - ROI pixels (after downsample): {info.roi_pixels_after_downsample}")
    print(f" - ROI pixels used in this run: {info.roi_pixels_used}")
    if info.sampled:
        print(f" - sampled to max_pixels={info.max_pixels} with seed={args.seed}")

    print(f"Dendrogram saved: {dendro_path}")
    print("Outputs written to:", os.path.abspath(args.out))


def build_parser():
    parser = argparse.ArgumentParser(description="ROI-based HCA clustering for aligned images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_select = subparsers.add_parser("select-roi", help="Select ROI on a reference image.")
    p_select.add_argument("--image", required=True, help="Path to reference image.")
    p_select.add_argument("--out", required=True, help="Output directory for ROI files.")
    p_select.add_argument("--notes", default="", help="Optional notes to store in roi.json.")
    p_select.set_defaults(func=cmd_select_roi)

    p_cluster = subparsers.add_parser("cluster", help="Cluster images using ROI pixels.")
    p_cluster.add_argument("--input_dir", required=True, help="Directory with PNG/JPG images.")
    p_cluster.add_argument("--roi", required=True, help="Path to roi.json")
    p_cluster.add_argument("--out", required=True, help="Output directory.")
    p_cluster.add_argument("--metric", default="auto", choices=["euclidean", "cosine", "auto"])
    p_cluster.add_argument(
        "--metric_threshold",
        type=float,
        default=0.02,
        help="CCC improvement threshold for choosing cosine in auto mode.",
    )
    p_cluster.add_argument(
        "--linkage",
        default="complete",
        choices=["average", "complete", "single", "ward"],
    )
    p_cluster.add_argument(
        "--pairwise_complete_obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pairwise-complete observations when computing distances.",
    )
    p_cluster.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Bootstrap replicate count for cluster support estimation (0 disables).",
    )
    p_cluster.add_argument(
        "--bootstrap_seed",
        type=int,
        default=123,
        help="Random seed for bootstrap resampling.",
    )
    p_cluster.add_argument("--downsample_factor", type=int, default=1)
    p_cluster.add_argument("--max_pixels", type=int, default=None)
    p_cluster.add_argument("--seed", type=int, default=123)
    p_cluster.add_argument("--pca_components", type=int, default=None)
    p_cluster.add_argument(
        "--leaf_spacing",
        type=float,
        default=0.12,
        help="Figure width in inches per leaf for dendrogram spacing.",
    )
    p_cluster.add_argument(
        "--max_fig_width",
        type=float,
        default=140.0,
        help="Upper bound for dendrogram figure width in inches.",
    )
    p_cluster.add_argument("--n_clusters", type=int, default=None)
    p_cluster.add_argument("--report_html", action="store_true")
    p_cluster.set_defaults(func=cmd_cluster)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
