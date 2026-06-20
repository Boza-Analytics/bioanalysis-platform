"""Import SynthMT's data-generation API without triggering its heavy package
__init__ (which eagerly imports the benchmark zoo → seaborn/torch/µSAM).

We register a lightweight stub for the top-level `synth_mt` package whose
__path__ points at the real source dir, so `synth_mt.config.*` and
`synth_mt.data_generation.*` resolve to the real (CPU-only) modules while the
real package __init__ never runs.
"""

from __future__ import annotations

import os
import sys
import types

_THIS = os.path.dirname(os.path.abspath(__file__))
_SYNTHMT_PKG = os.path.abspath(
    os.path.join(_THIS, "..", "submodules", "SynthMT", "synth_mt")
)


def bootstrap() -> None:
    """Make `import synth_mt.<sub>` work using only generation-light deps."""
    if not os.path.isdir(_SYNTHMT_PKG):
        raise FileNotFoundError(
            f"SynthMT not found at {_SYNTHMT_PKG}. Clone it into submodules/."
        )
    if "synth_mt" not in sys.modules:
        pkg = types.ModuleType("synth_mt")
        pkg.__path__ = [_SYNTHMT_PKG]  # namespace-style: resolve real submodules
        sys.modules["synth_mt"] = pkg


bootstrap()
