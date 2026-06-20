# TNT training / tuning toolkit (SynthMT-driven)

The MVP goal: make the model detect **tunneling nanotubes (TNTs)** with some
accuracy, using **synthetic data only** (no real images required). Generic modes
keep using the pre-trained MedSAM3 weights.

## Why this works
Microtubules ≈ TNTs morphologically (thin, low-contrast, curvilinear filaments).
SynthMT generates realistic filament images **with perfect ground-truth masks**,
so we can tune and validate TNT detection with zero human annotation. Per the
SynthMT paper, text-prompted SAM3 reaches human-grade filament segmentation
after **hyperparameter optimization on a handful of synthetic images** — no
weight training needed. That's the cheap path we take first.

## Pipeline
```
1. Generate   python train/gen_synthetic_tnt.py --series 8 --keep 3 --out train/data/tnt
2. Tune (GPU) python train/hpo_tnt.py   --data train/data/tnt --trials 40 --subset 12
3. Eval (GPU) python train/eval_tnt.py  --data train/data/tnt --limit 12
```
- **Step 1** runs anywhere (CPU only). Writes `images/`, `masks/` (instance ids),
  `overlays/` (GT), and `manifest.json` (per-filament lengths in px/µm).
- **Step 2** searches prompt / threshold / NMS / Frangi scale against GT masks and
  writes the winners into `../configs/tnt_default.json`.
- **Step 3** reports foreground Dice, detection-count error, and length MAE (µm),
  and saves model-vs-GT overlays.

## Files
- `tnt_synth_config.json` — SynthMT generator params tuned for TNT morphology
  (sparser, straighter, bright-on-dark fluorescence look).
- `lib_synthmt.py` — imports SynthMT's CPU generation API without its heavy
  benchmark `__init__` (avoids torch/seaborn/µSAM at gen time).
- `gen_synthetic_tnt.py` / `hpo_tnt.py` / `eval_tnt.py` — the three steps above.
- `data/tnt/` — generated dataset; doubles as the frontend demo gallery.

## Honest scope
Everything is validated against **synthetic** ground truth we defined. This
proves the MVP capability on controlled data; it is **not** a clinically
validated TNT assay. When real TNT micrographs exist, plug them into SynthMT's
DINOv2 parameter alignment + a few-shot LoRA fine-tune (`train_sam3_lora.py` in
MedSAM3) to close the sim-to-real gap.
