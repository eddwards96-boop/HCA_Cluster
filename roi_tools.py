import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from matplotlib.patches import Polygon


@dataclass
class ROIData:
    image_width: int
    image_height: int
    polygon: list
    created_utc: str
    notes: str = ""


class _ROISelector:
    def __init__(self, image_array):
        self.image_array = image_array
        self.verts = None
        self.cancelled = False
        self.done = False
        self._poly_patch = None

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image_array)
        self.ax.set_title("Draw ROI (lasso). Enter=confirm, r/Esc=reset, q=cancel")
        self.ax.set_axis_off()

        self.lasso = LassoSelector(self.ax, onselect=self._on_select)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.cid_close = self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_select(self, verts):
        self.verts = verts
        if self._poly_patch is not None:
            self._poly_patch.remove()
            self._poly_patch = None
        self._poly_patch = Polygon(verts, closed=True, fill=False, edgecolor="red", linewidth=2)
        self.ax.add_patch(self._poly_patch)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("enter", "return"):
            if self.verts is not None and len(self.verts) >= 3:
                self.done = True
                plt.close(self.fig)
        elif event.key in ("escape", "r"):
            self.verts = None
            if self._poly_patch is not None:
                self._poly_patch.remove()
                self._poly_patch = None
            self.fig.canvas.draw_idle()
        elif event.key == "q":
            self.cancelled = True
            plt.close(self.fig)

    def _on_close(self, _event):
        if not self.done:
            self.cancelled = True


def select_roi_interactive(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    img = img.convert("RGB")
    image_array = np.asarray(img)

    selector = _ROISelector(image_array)
    plt.show()

    if selector.cancelled or selector.verts is None:
        return None, image_array.shape[1], image_array.shape[0]

    return selector.verts, image_array.shape[1], image_array.shape[0]


def polygon_to_mask(vertices, width, height):
    if not vertices or len(vertices) < 3:
        return np.zeros((height, width), dtype=bool)

    poly_path = Path(vertices)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.vstack((x.ravel() + 0.5, y.ravel() + 0.5)).T
    mask = poly_path.contains_points(points)
    return mask.reshape((height, width))


def save_roi(out_dir, width, height, vertices, notes=""):
    os.makedirs(out_dir, exist_ok=True)

    roi_data = ROIData(
        image_width=int(width),
        image_height=int(height),
        polygon=[{"x": float(v[0]), "y": float(v[1])} for v in vertices],
        created_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        notes=notes or "",
    )

    roi_json = os.path.join(out_dir, "roi.json")
    with open(roi_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_width": roi_data.image_width,
                "image_height": roi_data.image_height,
                "roi_type": "polygon",
                "polygon": roi_data.polygon,
                "created_utc": roi_data.created_utc,
                "notes": roi_data.notes,
            },
            f,
            indent=2,
        )

    mask = polygon_to_mask(vertices, width, height)
    if mask.sum() == 0:
        raise ValueError("Empty ROI: mask has 0 pixels.")

    mask_path = os.path.join(out_dir, "roi_mask.png")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img.save(mask_path)

    return roi_json, mask_path


def load_roi(roi_json_path):
    if not os.path.isfile(roi_json_path):
        raise FileNotFoundError(f"ROI json not found: {roi_json_path}")

    with open(roi_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["image_width", "image_height", "roi_type", "polygon"]
    for key in required:
        if key not in data:
            raise ValueError(f"ROI json missing required key: {key}")

    if data.get("roi_type") != "polygon":
        raise ValueError("ROI type must be 'polygon'.")

    base_dir = os.path.dirname(os.path.abspath(roi_json_path))
    mask_path = os.path.join(base_dir, "roi_mask.png")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")

    mask_img = Image.open(mask_path).convert("L")
    mask = np.asarray(mask_img) > 0

    width = int(data["image_width"])
    height = int(data["image_height"])
    if mask.shape[1] != width or mask.shape[0] != height:
        raise ValueError(
            f"ROI mask size {mask.shape[1]}x{mask.shape[0]} does not match roi.json "
            f"{width}x{height}"
        )

    return data, mask

