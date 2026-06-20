"""Shared in-process MedSAM3 (SAM3 + LoRA) runner.

The upstream `submodules/MedSAM3/infer_sam.py` is a *visualization* CLI: it
writes an annotated PNG and prints a summary, but it does not emit masks or
scores in any machine-readable form, and running it as a subprocess would
reload the multi-GB SAM3 model on every request.

This module instead imports the `SAM3LoRAInference` class directly and keeps a
single instance alive (singleton), so:

  * the model is loaded exactly once at startup and kept in memory, and
  * `predict()` returns real mask numpy arrays + scores that the TNT/PLA
    pipelines can skeletonize / blob-detect on.

Per-request `threshold` and `nms_iou` are applied by mutating the inference
instance's attributes before each call (they are read at predict time inside
`SAM3LoRAInference.predict`), so no rebuild is needed.

The MedSAM3 code uses cwd-relative asset paths (e.g.
`sam3/assets/bpe_simple_vocab_16e6.txt.gz`) and top-level imports
(`from sam3.model_builder import ...`, `from lora_layers import ...`), so we
add its directory to `sys.path` and `chdir` into it around model build and
inference.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

# --- Paths -----------------------------------------------------------------
PIPELINES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINES_DIR.parent
MEDSAM3_DIR = PROJECT_ROOT / "submodules" / "MedSAM3"
DEFAULT_CONFIG = MEDSAM3_DIR / "configs" / "full_lora_config.yaml"
DEFAULT_WEIGHTS = MEDSAM3_DIR / "weights" / "medsam3_v1" / "best_lora_weights.pt"


@contextlib.contextmanager
def _pushd(path: Path):
    """Temporarily chdir into `path` (MedSAM3 uses cwd-relative asset paths)."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


class MedSAM3Runner:
    """Thin wrapper around `SAM3LoRAInference` returning structured results."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        weights_path: Path = DEFAULT_WEIGHTS,
        device: str = "cuda",
        resolution: int = 1008,
    ):
        if not MEDSAM3_DIR.exists():
            raise FileNotFoundError(
                f"MedSAM3 not found at {MEDSAM3_DIR}. "
                "Did you clone it into submodules/ and run pip install -e .?"
            )
        if str(MEDSAM3_DIR) not in sys.path:
            sys.path.insert(0, str(MEDSAM3_DIR))

        # Build the model inside the MedSAM3 dir so relative asset paths resolve.
        with _pushd(MEDSAM3_DIR):
            from infer_sam import SAM3LoRAInference  # noqa: WPS433 (local import by design)

            self._inf = SAM3LoRAInference(
                config_path=str(config_path),
                weights_path=str(weights_path),
                resolution=resolution,
                device=device,
            )
        self._lock = threading.Lock()

    def predict(
        self,
        image_path: str,
        prompt: str,
        threshold: float = 0.5,
        nms_iou: float = 0.5,
    ) -> dict:
        """Run text-prompted segmentation on one image with one prompt.

        Returns a JSON-friendly dict:
            {
                "prompt": str,
                "num_detections": int,
                "scores": list[float],
                "boxes": list[[x1, y1, x2, y2]],   # original-image pixels
                "masks": list[np.ndarray(bool, HxW)]   # original-image size
            }
        """
        abs_image = os.path.abspath(image_path)
        if not os.path.exists(abs_image):
            raise FileNotFoundError(f"Image not found: {abs_image}")

        # SAM3LoRAInference reads these attributes at predict time.
        with self._lock:  # the model is not re-entrant; serialize requests
            self._inf.detection_threshold = float(threshold)
            self._inf.nms_iou_threshold = float(nms_iou)
            with _pushd(MEDSAM3_DIR):
                raw = self._inf.predict(abs_image, [prompt])

        res = raw.get(0, {})
        masks = res.get("masks")
        scores = res.get("scores")
        boxes = res.get("boxes")
        num = int(res.get("num_detections", 0) or 0)

        return {
            "prompt": prompt,
            "num_detections": num,
            "scores": [] if scores is None else [float(s) for s in np.asarray(scores).ravel()],
            "boxes": [] if boxes is None else np.asarray(boxes).reshape(-1, 4).tolist(),
            # split [N, H, W] into a list of bool HxW arrays for easy iteration
            "masks": [] if masks is None else [np.asarray(m).astype(bool) for m in np.asarray(masks)],
        }


# --- Singleton management --------------------------------------------------
_RUNNER: Optional[MedSAM3Runner] = None
_RUNNER_LOCK = threading.Lock()


def get_runner(
    config_path: Path = DEFAULT_CONFIG,
    weights_path: Path = DEFAULT_WEIGHTS,
    device: str = "cuda",
) -> MedSAM3Runner:
    """Return the process-wide MedSAM3 runner, building it on first use."""
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = MedSAM3Runner(
                    config_path=config_path,
                    weights_path=weights_path,
                    device=device,
                )
    return _RUNNER


def is_loaded() -> bool:
    """True if the model has been built in this process."""
    return _RUNNER is not None


def gpu_available() -> bool:
    """True if a CUDA device is visible to torch (best-effort, import-safe)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
