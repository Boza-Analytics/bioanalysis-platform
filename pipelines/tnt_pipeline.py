"""Tunneling-nanotube (TNT) detection + measurement pipeline.

Flow:
    1. Frangi vesselness filter to highlight thin filaments.
    2. Text-prompted SAM3+LoRA segmentation ("thin line") on the Frangi image.
    3. Skeletonize each returned mask.
    4. Measure length (regionprops major_axis_length) and width (distance
       transform mean along the skeleton) for each filament.
    5. Save a magenta-skeleton overlay on the original image.

Standalone:
    python pipelines/tnt_pipeline.py path/to/image.png [--pixel-size 0.065]
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
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.measure import label, regionprops
from skimage.morphology import dilation, skeletonize

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "tnt_default.json"


def _load_config() -> dict:
    defaults = {
        "use_frangi": False,
        "frangi_sigmas": [1, 2, 3, 4],
        "sam_threshold": 0.35,
        "nms_iou": 0.5,
        "sam_prompt": "thin line",
        "min_tnt_length_px": 25,
        "max_tnt_length_px": 1200,
        "straightness_min": 0.6,
        "border_margin_px": 0,
    }
    try:
        with open(CONFIG_PATH) as f:
            defaults.update(json.load(f))
    except FileNotFoundError:
        pass
    return defaults


def _to_gray_float(image_path: str) -> np.ndarray:
    """Load image as a float grayscale array in [0, 1]."""
    img = Image.open(image_path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def _touches_border(mask: np.ndarray, margin: int) -> bool:
    if margin <= 0:
        return False
    m = mask
    return bool(
        m[:margin, :].any()
        or m[-margin:, :].any()
        or m[:, :margin].any()
        or m[:, -margin:].any()
    )


def _measure_skeleton(skel: np.ndarray, mask: np.ndarray) -> dict:
    """Return length (px), width (px) and straightness for one skeletonized mask."""
    skel_pixels = int(skel.sum())
    if skel_pixels == 0:
        return {"length_px": 0.0, "width_px": 0.0, "straightness": 0.0}

    # Length: ellipse-fit major axis of the skeleton region.
    props = regionprops(label(skel.astype(np.uint8)))
    major = max((p.axis_major_length for p in props), default=0.0)
    # A 1px-wide line has major_axis_length ~ extent; use the skeleton pixel
    # count as a lower bound so single-blob skeletons still register a length.
    length_px = float(max(major, skel_pixels - 1))

    # Width: EDT on the mask gives, at each centerline pixel, the distance to
    # the nearest background pixel (~ the half-width). Full width ~ 2x mean.
    edt = ndi.distance_transform_edt(mask)
    radius_mean = float(edt[skel].mean()) if skel.any() else 0.0
    width_px = 2.0 * radius_mean

    # Straightness: end-to-end extent vs. traced length (1.0 == perfectly straight).
    straightness = float(min(1.0, major / skel_pixels)) if skel_pixels > 0 else 0.0
    return {"length_px": length_px, "width_px": width_px, "straightness": straightness}


def _save_overlay(image_path: str, skeletons: list[np.ndarray]) -> str:
    """Original image with all skeletons drawn in magenta -> /tmp PNG."""
    base = np.asarray(Image.open(image_path).convert("RGB")).copy()
    if skeletons:
        combined = np.zeros(base.shape[:2], dtype=bool)
        for skel in skeletons:
            combined |= skel
        # Dilate a touch so 1px skeletons are visible in the PNG.
        combined = dilation(combined)
        base[combined] = [255, 0, 255]  # magenta
    out_path = os.path.join(tempfile.gettempdir(), f"tnt_overlay_{uuid.uuid4().hex}.png")
    Image.fromarray(base).save(out_path)
    return out_path


def run_tnt(
    image_path: str,
    pixel_size_um: Optional[float] = None,
    runner=None,
    config: Optional[dict] = None,
) -> dict:
    """Detect and measure tunneling nanotubes in `image_path`.

    Returns:
        {
            "count": int,
            "lengths_um": list[float] | None,
            "lengths_px": list[float],
            "mean_width_px": float,
            "overlay_path": str,
        }
    """
    cfg = config or _load_config()

    # 1. Optionally Frangi-enhance thin filaments before SAM.
    #    On REAL micrographs Frangi turns the image into a ridge map that strips
    #    away context and makes SAM segment noise ("random lines"), so it defaults
    #    OFF — SAM3 runs directly on the original image. It helps mainly on the
    #    clean synthetic data, where it can be re-enabled via use_frangi=true.
    if cfg.get("use_frangi", False):
        gray = _to_gray_float(image_path)
        ridges = frangi(gray, sigmas=cfg["frangi_sigmas"], black_ridges=False)
        ridges_u8 = (np.clip(ridges, 0.0, 1.0) * 255).astype(np.uint8)
        sam_input = os.path.join(tempfile.gettempdir(), f"tnt_frangi_{uuid.uuid4().hex}.png")
        Image.fromarray(ridges_u8).convert("RGB").save(sam_input)
    else:
        sam_input = image_path

    # 2. SAM3 + LoRA text-prompted segmentation.
    if runner is None:
        from pipelines.medsam3_runner import get_runner

        runner = get_runner()
    seg = runner.predict(
        sam_input,
        prompt=cfg.get("sam_prompt", "thin line"),
        threshold=cfg["sam_threshold"],
        nms_iou=cfg["nms_iou"],
    )

    # 3-4. Skeletonize + measure each mask, applying config filters.
    lengths_px: list[float] = []
    widths_px: list[float] = []
    kept_skeletons: list[np.ndarray] = []

    for mask in seg["masks"]:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        if _touches_border(mask, int(cfg.get("border_margin_px", 0))):
            continue
        skel = skeletonize(mask)
        m = _measure_skeleton(skel, mask)
        if m["length_px"] < cfg["min_tnt_length_px"]:
            continue
        if m["length_px"] > cfg["max_tnt_length_px"]:
            continue
        if m["straightness"] < cfg["straightness_min"]:
            continue
        lengths_px.append(round(m["length_px"], 2))
        widths_px.append(m["width_px"])
        kept_skeletons.append(skel)

    # 5. Overlay (handles the zero-detection case gracefully).
    overlay_path = _save_overlay(image_path, kept_skeletons)

    mean_width_px = round(float(np.mean(widths_px)), 3) if widths_px else 0.0
    lengths_um = (
        [round(l * pixel_size_um, 3) for l in lengths_px]
        if pixel_size_um is not None
        else None
    )

    return {
        "count": len(lengths_px),
        "lengths_um": lengths_um,
        "lengths_px": lengths_px,
        "mean_width_px": mean_width_px,
        "overlay_path": overlay_path,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the TNT pipeline on one image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        dest="pixel_size_um",
        help="Pixel size in micrometres per pixel (enables µm output)",
    )
    args = parser.parse_args()
    result = run_tnt(args.image, pixel_size_um=args.pixel_size_um)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
