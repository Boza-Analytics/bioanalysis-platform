"""Hyperparameter optimization for TNT detection against synthetic ground truth.

This is SynthMT's core recipe: rather than train weights, search SAM3's
inference parameters (text prompt, detection threshold, NMS IoU) plus our Frangi
pre-filter scale against synthetic images that have perfect GT masks. The best
config is written to configs/tnt_default.json.

Runs on the GPU box (needs the in-process MedSAM3 model).

Usage (from project root, in the box venv):
    python train/hpo_tnt.py --data train/data/tnt --trials 40 --subset 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid

import imageio.v2 as imageio
import numpy as np
import optuna
from skimage.filters import frangi

# Make `pipelines` importable when run from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines.medsam3_runner import get_runner  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_OUT = os.path.join(PROJECT_ROOT, "configs", "tnt_default.json")

PROMPTS = [
    "thin line",
    "tunneling nanotube",
    "thin bright filament",
    "filament",
    "membrane nanotube",
]


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def _predict_union(runner, image_path, prompt, threshold, nms_iou, sigmas):
    """Frangi -> SAM3 -> union of predicted masks (bool HxW)."""
    gray = imageio.imread(image_path)
    if gray.ndim == 3:
        gray = gray[..., :3].mean(-1)
    gray = gray.astype(np.float64) / 255.0
    ridges = frangi(gray, sigmas=sigmas, black_ridges=False)
    tmp = os.path.join(tempfile.gettempdir(), f"hpo_{uuid.uuid4().hex}.png")
    imageio.imwrite(tmp, (np.clip(ridges, 0, 1) * 255).astype(np.uint8))
    try:
        seg = runner.predict(tmp, prompt=prompt, threshold=threshold, nms_iou=nms_iou)
    finally:
        os.remove(tmp)
    h, w = gray.shape
    union = np.zeros((h, w), bool)
    for m in seg["masks"]:
        union |= np.asarray(m, bool)
    return union, seg["num_detections"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "train", "data", "tnt"))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--subset", type=int, default=12, help="images scored per trial")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(manifest))[: args.subset]
    items = [manifest[i] for i in idx]

    # Pre-load images + GT unions.
    samples = []
    for it in items:
        img = os.path.join(args.data, it["image"])
        gt = imageio.imread(os.path.join(args.data, it["mask"])) > 0
        samples.append((img, gt, it["n_filaments"]))

    runner = get_runner()  # builds SAM3 once

    def objective(trial: optuna.Trial) -> float:
        prompt = trial.suggest_categorical("sam_prompt", PROMPTS)
        threshold = trial.suggest_float("sam_threshold", 0.10, 0.60, step=0.05)
        nms_iou = trial.suggest_float("nms_iou", 0.30, 0.70, step=0.10)
        max_sigma = trial.suggest_int("frangi_max_sigma", 3, 6)
        sigmas = list(range(1, max_sigma + 1))

        dices, count_err = [], []
        for img, gt, n_gt in samples:
            pred, n_det = _predict_union(runner, img, prompt, threshold, nms_iou, sigmas)
            dices.append(_dice(pred, gt))
            count_err.append(abs(n_det - n_gt) / max(n_gt, 1))
        # reward overlap, lightly penalize wrong detection counts
        score = float(np.mean(dices)) - 0.1 * float(np.mean(count_err))
        trial.set_user_attr("mean_dice", float(np.mean(dices)))
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)

    best = study.best_params
    best_sigmas = list(range(1, best["frangi_max_sigma"] + 1))
    print("\n=== BEST ===")
    print("params:", best)
    print("mean_dice:", study.best_trial.user_attrs.get("mean_dice"))

    cfg = json.load(open(CONFIG_OUT))
    cfg["sam_prompt"] = best["sam_prompt"]
    cfg["sam_threshold"] = round(best["sam_threshold"], 3)
    cfg["nms_iou"] = round(best["nms_iou"], 3)
    cfg["frangi_sigmas"] = best_sigmas
    with open(CONFIG_OUT, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote tuned config -> {CONFIG_OUT}")


if __name__ == "__main__":
    main()
