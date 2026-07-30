import os
import platform
from typing import Any

# Windows can load both libomp.dll and libiomp5md.dll through PyTorch/OpenCV/NumPy.
# Set this before importing the pipeline modules.
if platform.system() == "Windows":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__all__ = ["main", "process_building"]


def __getattr__(name: str) -> Any:
    """Load the GPU-heavy pipeline only when a public entry point is requested."""
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
