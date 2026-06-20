"""Evaluate TNT detection (current configs/tnt_default.json) on synthetic GT.

Reports foreground Dice, detection-count error, and filament length MAE (µm),
and writes model-vs-ground-truth comparison overlays for the demo gallery.

Runs on the GPU box. Usage:
    python train/eval_tnt.py --data train/data/tnt --limit 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines.tnt_pipeline import _load_config, run_tnt  # noqa: E402
from pipelines.medsam3_runner import get_runner  # noqa: E402
from hpo_tnt import _dice, _predict_union  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _length_mae(pred_um, gt_um):
    """Greedy length matching MAE (µm) over the min count; unmatched ignored."""
    if not pred_um or not gt_um:
        return None
    p = sorted(pred_um, reverse=True)
    g = sorted(gt_um, reverse=True)
    k = min(len(p), len(g))
    return float(np.mean([abs(p[i] - g[i]) for i in range(k)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "train", "data", "tnt"))
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))[: args.limit]
    cfg = _load_config()
    runner = get_runner()
    out_dir = os.path.join(args.data, "eval_overlays")
    os.makedirs(out_dir, exist_ok=True)

    dices, count_errs, len_maes, rows = [], [], [], []
    for it in manifest:
        img = os.path.join(args.data, it["image"])
        gt = imageio.imread(os.path.join(args.data, it["mask"])) > 0

        pred_union, n_det = _predict_union(
            runner, img, cfg["sam_prompt"], cfg["sam_threshold"],
            cfg["nms_iou"], cfg["frangi_sigmas"],
        )
        d = _dice(pred_union, gt)
        res = run_tnt(img, pixel_size_um=it["um_per_pixel"], runner=runner, config=cfg)
        cerr = abs(res["count"] - it["n_filaments"])
        lmae = _length_mae(res.get("lengths_um") or [], it["lengths_um"])

        dices.append(d)
        count_errs.append(cerr)
        if lmae is not None:
            len_maes.append(lmae)
        rows.append({
            "id": it["id"], "dice": round(d, 3),
            "n_gt": it["n_filaments"], "n_pred": res["count"],
            "length_mae_um": None if lmae is None else round(lmae, 2),
        })
        # save the model overlay next to GT for visual comparison
        if os.path.exists(res["overlay_path"]):
            imageio.imwrite(
                os.path.join(out_dir, f"{it['id']}_pred.png"),
                imageio.imread(res["overlay_path"]),
            )

    report = {
        "n_images": len(manifest),
        "mean_dice": round(float(np.mean(dices)), 3),
        "mean_count_abs_err": round(float(np.mean(count_errs)), 2),
        "length_mae_um": round(float(np.mean(len_maes)), 2) if len_maes else None,
        "config": cfg,
        "per_image": rows,
    }
    with open(os.path.join(args.data, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ["n_images", "mean_dice", "mean_count_abs_err", "length_mae_um"]}, indent=2))
    print(f"\nreport -> {os.path.join(args.data, 'eval_report.json')}")
    print(f"overlays -> {out_dir}")


if __name__ == "__main__":
    main()
