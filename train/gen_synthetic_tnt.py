"""Generate a synthetic TNT-like dataset (images + instance masks + GT lengths)
using SynthMT's microtubule generator, tuned for tunneling-nanotube morphology.

Why microtubules ≈ TNTs: both are thin, low-contrast, curvilinear filaments.
SynthMT gives us unlimited images WITH perfect ground-truth masks (no human
labels), which is exactly what we need to (a) tune/validate TNT detection and
(b) build a labeled demo gallery.

Output layout (under --out):
    images/   <id>.png            RGB synthetic micrograph
    masks/    <id>.png            uint16 foreground (union) mask, 0/instance-id
    overlays/ <id>.png            GT overlay (filaments outlined) for demos
    manifest.json                 per-image: n_filaments, per-filament length px/µm

Usage:
    python train/gen_synthetic_tnt.py --series 8 --keep 3 --out train/data/tnt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile

import lib_synthmt  # noqa: F401 — registers the light synth_mt stub (import first)
import imageio.v2 as imageio
import numpy as np
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

from synth_mt.config.synthetic_data import SyntheticDataConfig
from synth_mt.data_generation.video import generate_video

THIS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(THIS, "tnt_synth_config.json")


def _instance_length_px(plane: np.ndarray) -> float:
    """Arc length of one filament ≈ skeleton pixel count (good for curved lines)."""
    skel = skeletonize(plane > 0)
    n = int(skel.sum())
    return float(max(n - 1, 0))


def _gt_overlay(image: np.ndarray, stack: np.ndarray) -> np.ndarray:
    """Draw GT filament boundaries (magenta) over the image for demo display."""
    out = image.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    union = (stack > 0).any(axis=0)
    border = find_boundaries(union, mode="outer")
    out[border] = [255, 0, 255]
    return out


def _randomize(cfg: SyntheticDataConfig, rng: random.Random) -> None:
    """Light domain randomization across series for dataset diversity."""
    cfg.num_microtubule = rng.randint(4, 8)
    cfg.background_level = round(rng.uniform(0.06, 0.16), 3)
    cfg.tubulus_contrast = round(rng.uniform(0.40, 0.70), 3)
    cfg.gaussian_noise = round(rng.uniform(0.02, 0.05), 3)
    cfg.global_blur_sigma = round(rng.uniform(0.5, 0.9), 3)
    cfg.bending_angle_gamma_scale = round(rng.uniform(0.003, 0.008), 4)
    cfg.random_spots.count = rng.randint(20, 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic TNT dataset.")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--series", type=int, default=8, help="number of independent series")
    ap.add_argument("--keep", type=int, default=3, help="frames kept per series (latest, spaced)")
    ap.add_argument("--out", default=os.path.join(THIS, "data", "tnt"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for sub in ("images", "masks", "overlays"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    manifest = []
    raw_root = tempfile.mkdtemp(prefix="synthmt_raw_")
    try:
        for s in range(args.series):
            cfg = SyntheticDataConfig.from_json(args.config)
            cfg.id = f"tnt{s:03d}"
            _randomize(cfg, rng)
            um = float(cfg.um_per_pixel)

            raw = os.path.join(raw_root, cfg.id)
            os.makedirs(raw, exist_ok=True)
            generate_video(cfg, raw)

            img_dir = os.path.join(raw, "full", "images")
            msk_dir = os.path.join(raw, "full", "image_masks")
            frames = sorted(os.listdir(img_dir))

            # Keep the latest `keep` frames (filaments fully grown), spaced out.
            n = len(frames)
            kept_idx = sorted(set(
                max(0, n - 1 - round(i * (n / max(args.keep, 1)) / 2))
                for i in range(args.keep)
            ))
            for fi in kept_idx:
                fname = frames[fi]
                stem = os.path.splitext(fname)[0]
                image = imageio.imread(os.path.join(img_dir, fname))
                stack = imageio.imread(os.path.join(msk_dir, stem + ".tif"))
                if stack.ndim == 2:
                    stack = stack[None]

                uid = f"{cfg.id}_{stem.split('_')[-1]}"
                imageio.imwrite(os.path.join(args.out, "images", uid + ".png"), image)
                union = (stack > 0).any(axis=0).astype(np.uint16)
                # encode instance ids in the union mask where possible
                inst = np.zeros(union.shape, np.uint16)
                for i in range(stack.shape[0]):
                    inst[stack[i] > 0] = i + 1
                imageio.imwrite(os.path.join(args.out, "masks", uid + ".png"), inst)
                imageio.imwrite(
                    os.path.join(args.out, "overlays", uid + ".png"),
                    _gt_overlay(image, stack),
                )

                lengths_px = [
                    _instance_length_px(stack[i])
                    for i in range(stack.shape[0])
                    if (stack[i] > 0).sum() > 5
                ]
                manifest.append({
                    "id": uid,
                    "image": f"images/{uid}.png",
                    "mask": f"masks/{uid}.png",
                    "overlay": f"overlays/{uid}.png",
                    "um_per_pixel": um,
                    "n_filaments": len(lengths_px),
                    "lengths_px": [round(x, 1) for x in lengths_px],
                    "lengths_um": [round(x * um, 3) for x in lengths_px],
                })
            print(f"series {cfg.id}: kept {len(kept_idx)} frames")
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total_fil = sum(m["n_filaments"] for m in manifest)
    print(f"\nDONE: {len(manifest)} images, {total_fil} filaments -> {args.out}")
    print(f"manifest: {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
