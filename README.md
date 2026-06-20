# BioAnalysis Platform (MVP)

Text-promptable biomedical image analysis built on **MedSAM3** (SAM3 + LoRA).
Three modes:

| Mode | What it does |
|------|--------------|
| **TNT** | Detects & measures tunneling-nanotube filaments (Frangi → SAM3 → skeleton → length/width) |
| **PLA** | Counts proximity-ligation-assay spots per cell (SAM3 cell ROIs → LoG blob detection) |
| **General** | Free-text prompted segmentation over MedSAM3's medical concepts |

```
bioanalysis-platform/
├── submodules/
│   ├── MedSAM3/          # https://github.com/Joey-S-Liu/MedSAM3  (SAM3 + LoRA)
│   └── SynthMT/          # https://github.com/ml-lab-htw/SynthMT  (installed; not yet wired in)
├── pipelines/
│   ├── medsam3_runner.py # shared in-process SAM3 singleton  ← see "Key design decision"
│   ├── tnt_pipeline.py   # run_tnt(image_path, pixel_size_um=None)
│   └── pla_pipeline.py   # run_pla(image_path)
├── api/server.py         # FastAPI: /analyse, /health, /modes
├── frontend/index.html   # single-file vanilla-JS UI (Tailwind CDN)
├── configs/{tnt,pla}_default.json
├── scripts/setup_ec2.sh  # unattended Ubuntu 22.04 provisioner
├── Dockerfile
└── requirements.txt
```

## Key design decision: in-process model, not subprocess

The original spec called for the pipelines to shell out to
`submodules/MedSAM3/infer_sam.py` per request. After reading the upstream code:

* `infer_sam.py` is a **visualization CLI** — it writes an annotated PNG and
  prints a summary, but **does not emit masks or scores** in any
  machine-readable form. The pipelines need the actual mask arrays to
  skeletonize / run blob detection.
* Shelling out reloads the multi-GB SAM3 model on **every** request, directly
  violating the constraint *"model loaded once at startup, kept in memory."*

So `pipelines/medsam3_runner.py` imports the `SAM3LoRAInference` class
in-process, builds it **once** as a singleton, and exposes
`predict(image_path, prompt, threshold, nms_iou)` returning
`{masks, scores, boxes, num_detections}`. Per-request thresholds are applied by
setting the inference instance's attributes (read at predict time) — no rebuild.
MedSAM3 uses cwd-relative asset paths, so the runner `chdir`s into the MedSAM3
directory around build and inference.

## Endpoints

* `POST /analyse` — multipart `file`, `mode` (`tnt|pla|general`), `prompt`
  (required for `general`), `pixel_size_um` (optional, TNT). Returns
  `{mode, result, overlay_image}` where `overlay_image` is base64 PNG.
* `GET /health` — `{status, gpu, medsam3_loaded}`
* `GET /modes` — mode descriptors for the UI.

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e submodules/MedSAM3          # heavy ML deps (torch, sam3, …)
# place LoRA weights at submodules/MedSAM3/weights/medsam3_v1/best_lora_weights.pt
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` directly in a browser, or serve it. The
frontend auto-targets `http://<host>:8000` (localhost or the EC2 public IP) — no
config change needed between dev and prod.

### Standalone pipeline testing

```bash
python pipelines/tnt_pipeline.py path/to/image.png --pixel-size 0.065
python pipelines/pla_pipeline.py path/to/image.png
```

## Deploy to EC2

```bash
# On a fresh Ubuntu 22.04 GPU instance (e.g. g5.xlarge), as the ubuntu user:
REPO_URL=<git-url> HF_TOKEN=<hf-token> bash scripts/setup_ec2.sh
```

This installs Python 3.11, Docker, nginx, the NVIDIA driver + CUDA 12.1 (only if
a GPU is present), clones both submodules, installs everything into a venv,
downloads the gated weights (`lal-Joey/MedSAM3_v1`), and registers
`bioanalysis.service` (uvicorn on :8000, `Restart=always`) plus an nginx reverse
proxy (`/` → frontend, `/api/` → :8000).

## Notes & limitations (MVP)

* **Weights are gated** on Hugging Face — `HF_TOKEN` is required for the
  unattended download. Without weights the API boots and serves `/health` /
  `/modes`, but `/analyse` returns **503** until the weights exist.
* No auth, no database, no S3 — all files are local to the instance (`/tmp`).
* CPU inference works but is very slow; a GPU instance is expected for real use.
* **SynthMT** is cloned and installed but not yet wired into any pipeline
  (kept available for future synthetic-data generation / fine-tuning).
* Requests are serialized through the model with a lock (SAM3 forward passes are
  not re-entrant); for higher throughput, run multiple workers behind nginx.
