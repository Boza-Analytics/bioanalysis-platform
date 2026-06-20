"""Proximity-ligation-assay (PLA) spot-counting pipeline.

Flow:
    1. Text-prompted SAM3+LoRA segmentation ("cell nucleus") to get cell ROIs.
    2. For each ROI, Laplacian-of-Gaussian blob detection (blob_log) restricted
       to the masked region; count spots whose centre falls inside the ROI.
    3. Save an overlay: cell outlines in green, spots as cyan dots.

Standalone:
    python pipelines/pla_pipeline.py path/to/image.png
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.morphology import erosion

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pla_default.json"


def _load_config() -> dict:
    defaults = {
        "sam_threshold": 0.5,
        "sam_nms_iou": 0.5,
        "blob_min_sigma": 1,
        "blob_max_sigma": 4,
        "blob_threshold": 0.1,
    }
    try:
        with open(CONFIG_PATH) as f:
            defaults.update(json.load(f))
    except FileNotFoundError:
        pass
    return defaults


def _to_gray_float(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def _detect_spots_in_mask(gray: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Return Nx2 array of (row, col) spot centres inside `mask`."""
    masked = gray * mask  # suppress signal outside the ROI
    blobs = blob_log(
        masked,
        min_sigma=cfg["blob_min_sigma"],
        max_sigma=cfg["blob_max_sigma"],
        threshold=cfg["blob_threshold"],
    )
    if blobs.size == 0:
        return np.empty((0, 2), dtype=int)
    centres = []
    h, w = mask.shape
    for y, x, _sigma in blobs:
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
            centres.append((yi, xi))
    return np.asarray(centres, dtype=int) if centres else np.empty((0, 2), dtype=int)


def _save_overlay(
    image_path: str,
    cell_masks: list[np.ndarray],
    all_spots: list[np.ndarray],
) -> str:
    """Cell outlines (green) + spots (cyan dots) on the original image -> /tmp PNG."""
    base = np.asarray(Image.open(image_path).convert("RGB")).copy()
    h, w = base.shape[:2]

    for mask in cell_masks:
        boundary = mask ^ erosion(mask)
        base[boundary] = [0, 255, 0]  # green outline

    for spots in all_spots:
        for yi, xi in spots:
            rr, cc = disk((yi, xi), radius=3, shape=(h, w))
            base[rr, cc] = [0, 255, 255]  # cyan dot

    out_path = os.path.join(tempfile.gettempdir(), f"pla_overlay_{uuid.uuid4().hex}.png")
    Image.fromarray(base).save(out_path)
    return out_path


def run_pla(image_path: str, runner=None, config: Optional[dict] = None) -> dict:
    """Count PLA spots per cell in `image_path`.

    Returns:
        {
            "cell_count": int,
            "spots_per_cell": list[int],
            "total_spots": int,
            "mean_spots": float,
            "overlay_path": str,
        }
    """
    cfg = config or _load_config()

    # 1. Cell ROI segmentation.
    if runner is None:
        from pipelines.medsam3_runner import get_runner

        runner = get_runner()
    seg = runner.predict(
        image_path,
        prompt="cell nucleus",
        threshold=cfg["sam_threshold"],
        nms_iou=cfg.get("sam_nms_iou", 0.5),
    )

    gray = _to_gray_float(image_path)

    # 2. Blob detection per ROI.
    cell_masks: list[np.ndarray] = []
    all_spots: list[np.ndarray] = []
    spots_per_cell: list[int] = []
    for mask in seg["masks"]:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        spots = _detect_spots_in_mask(gray, mask, cfg)
        cell_masks.append(mask)
        all_spots.append(spots)
        spots_per_cell.append(int(len(spots)))

    # 3. Overlay (safe when there are no cells).
    overlay_path = _save_overlay(image_path, cell_masks, all_spots)

    cell_count = len(spots_per_cell)
    total_spots = int(sum(spots_per_cell))
    mean_spots = round(total_spots / cell_count, 3) if cell_count > 0 else 0.0

    return {
        "cell_count": cell_count,
        "spots_per_cell": spots_per_cell,
        "total_spots": total_spots,
        "mean_spots": mean_spots,
        "overlay_path": overlay_path,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the PLA pipeline on one image.")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()
    result = run_pla(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
