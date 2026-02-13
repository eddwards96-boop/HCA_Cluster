import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None


@dataclass
class FeatureInfo:
    num_images: int
    roi_pixels_original: int
    roi_pixels_after_downsample: int
    roi_pixels_used: int
    sampled: bool
    max_pixels: int | None
    downsample_factor: int
    pca_components: int | None
    final_dim: int


def _list_image_files(input_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    exts = {".png", ".jpg", ".jpeg"}
    files = []
    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(os.path.join(input_dir, name))

    files.sort()
    if len(files) < 2:
        raise ValueError("Need at least 2 images to cluster.")
    return files


def _load_image_rgb(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
        img = img.convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def _downsample_mask(mask, factor):
    if factor == 1:
        return mask
    if factor <= 0:
        raise ValueError("downsample_factor must be >= 1")
    h, w = mask.shape
    new_w = max(1, w // factor)
    new_h = max(1, h // factor)
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = mask_img.resize((new_w, new_h), resample=Image.NEAREST)
    return np.asarray(mask_img) > 0


def _downsample_image(image, factor):
    if factor == 1:
        return image
    if factor <= 0:
        raise ValueError("downsample_factor must be >= 1")
    w, h = image.size
    new_w = max(1, w // factor)
    new_h = max(1, h // factor)
    return image.resize((new_w, new_h), resample=Image.BILINEAR)


def build_feature_matrix(
    input_dir,
    roi_mask,
    downsample_factor=1,
    max_pixels=None,
    seed=123,
    pca_components=None,
):
    image_paths = _list_image_files(input_dir)

    expected_h, expected_w = roi_mask.shape

    size_mismatch = []
    for path in image_paths:
        with Image.open(path) as img:
            if img.size != (expected_w, expected_h):
                size_mismatch.append((path, img.size))

    if size_mismatch:
        details = ", ".join([f"{os.path.basename(p)}={w}x{h}" for p, (w, h) in size_mismatch])
        raise ValueError(f"Image sizes do not match ROI size {expected_w}x{expected_h}: {details}")

    mask_ds = _downsample_mask(roi_mask, downsample_factor)
    if mask_ds.sum() == 0:
        raise ValueError("ROI mask is empty after downsampling.")

    roi_pixels_original = int(np.count_nonzero(roi_mask))
    roi_pixels_after_downsample = int(np.count_nonzero(mask_ds))

    roi_indices = np.flatnonzero(mask_ds.ravel())
    if roi_indices.size == 0:
        raise ValueError("ROI mask has 0 pixels.")

    sampled = False
    if max_pixels is not None and max_pixels > 0 and roi_indices.size > max_pixels:
        rng = np.random.default_rng(seed)
        roi_indices = rng.choice(roi_indices, size=max_pixels, replace=False)
        roi_indices = np.sort(roi_indices)
        sampled = True

    roi_pixels_used = int(roi_indices.size)

    features = []
    filenames = []
    for path in image_paths:
        img = _load_image_rgb(path)
        if downsample_factor != 1:
            img = _downsample_image(img, downsample_factor)
        arr = np.asarray(img, dtype=np.float32)
        flat = arr.reshape(-1, 3)
        roi_pixels = flat[roi_indices]
        vec = roi_pixels.reshape(-1)
        features.append(vec)
        filenames.append(os.path.basename(path))

    X = np.stack(features, axis=0)

    final_dim = X.shape[1]
    if pca_components is not None:
        if PCA is None:
            raise ImportError("scikit-learn is required for PCA.")
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
        if pca_components > max_components:
            raise ValueError(
                f"pca_components={pca_components} is greater than min(n_samples, n_features)={max_components}"
            )
        pca = PCA(n_components=pca_components, random_state=seed)
        X = pca.fit_transform(X)
        final_dim = X.shape[1]

    info = FeatureInfo(
        num_images=len(image_paths),
        roi_pixels_original=roi_pixels_original,
        roi_pixels_after_downsample=roi_pixels_after_downsample,
        roi_pixels_used=roi_pixels_used,
        sampled=sampled,
        max_pixels=max_pixels,
        downsample_factor=downsample_factor,
        pca_components=pca_components,
        final_dim=final_dim,
    )

    return X, filenames, image_paths, info, mask_ds
