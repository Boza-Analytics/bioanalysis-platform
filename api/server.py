"""BioAnalysis FastAPI server.

Endpoints:
    POST /analyse   multipart: file, mode (tnt|pla|general), prompt?, pixel_size_um?
    GET  /health    -> {status, gpu, medsam3_loaded}
    GET  /modes     -> mode descriptors for the UI

The MedSAM3 model is built once at startup (lifespan) and shared across all
requests via `pipelines.medsam3_runner` — see that module for why we import the
class in-process instead of shelling out to infer_sam.py per request.
"""

from __future__ import annotations

import base64
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from pipelines import medsam3_runner
from pipelines.pla_pipeline import run_pla
from pipelines.tnt_pipeline import run_tnt

UPLOAD_DIR = Path(tempfile.gettempdir()) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODES = [
    {
        "id": "tnt",
        "label": "Tunneling nanotubes",
        "description": "Detects and measures TNT filaments between cells",
    },
    {
        "id": "pla",
        "label": "PLA spot counting",
        "description": "Counts proximity ligation assay spots per cell",
    },
    {
        "id": "general",
        "label": "General medicine",
        "description": "Text-prompted segmentation, 330 medical concepts",
    },
]

_OVERLAY_COLORS = [
    (255, 0, 0),
    (0, 128, 255),
    (0, 200, 0),
    (255, 200, 0),
    (200, 0, 255),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the SAM3+LoRA model once at startup; tolerate failure for dev."""
    try:
        medsam3_runner.get_runner()
        print("✅ MedSAM3 loaded at startup.")
    except Exception as exc:  # noqa: BLE001 - keep /health and /modes alive
        print(f"⚠️  MedSAM3 not loaded at startup: {exc}")
    yield


app = FastAPI(title="BioAnalysis", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _general_overlay(image_path: str, masks: list[np.ndarray]) -> str:
    """Translucent coloured masks on the original image -> /tmp PNG path."""
    base = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32)
    for i, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        color = np.array(_OVERLAY_COLORS[i % len(_OVERLAY_COLORS)], dtype=np.float32)
        base[mask] = 0.55 * base[mask] + 0.45 * color
    out = os.path.join(tempfile.gettempdir(), f"general_overlay_{uuid.uuid4().hex}.png")
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(out)
    return out


def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(dest, "wb") as out:
        out.write(file.file.read())
    return str(dest)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "gpu": medsam3_runner.gpu_available(),
        "medsam3_loaded": medsam3_runner.is_loaded(),
    }


@app.get("/modes")
def modes() -> list:
    return MODES


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    mode: str = Form(...),
    prompt: Optional[str] = Form(None),
    pixel_size_um: Optional[str] = Form(None),
):
    if mode not in {"tnt", "pla", "general"}:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    if mode == "general" and not (prompt and prompt.strip()):
        raise HTTPException(status_code=400, detail="prompt is required for general mode")

    px_um: Optional[float] = None
    if pixel_size_um not in (None, ""):
        try:
            px_um = float(pixel_size_um)
        except ValueError:
            raise HTTPException(status_code=400, detail="pixel_size_um must be a number")

    image_path = _save_upload(file)

    try:
        if mode == "tnt":
            result = run_tnt(image_path, pixel_size_um=px_um)
            overlay_b64 = _encode_png(result["overlay_path"])
        elif mode == "pla":
            result = run_pla(image_path)
            overlay_b64 = _encode_png(result["overlay_path"])
        else:  # general
            runner = medsam3_runner.get_runner()
            seg = runner.predict(image_path, prompt=prompt.strip(), threshold=0.5)
            overlay_path = _general_overlay(image_path, seg["masks"])
            result = {
                "prompt": seg["prompt"],
                "num_detections": seg["num_detections"],
                "confidence_scores": [round(s, 4) for s in seg["scores"]],
                "overlay_path": overlay_path,
            }
            overlay_b64 = _encode_png(overlay_path)
    except FileNotFoundError as exc:
        # Most common cause: model weights missing / model not loaded.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return {"mode": mode, "result": result, "overlay_image": overlay_b64}
