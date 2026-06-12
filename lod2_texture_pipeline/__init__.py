import os
import platform

# Windows can load both libomp.dll and libiomp5md.dll through PyTorch/OpenCV/NumPy.
# Set this before importing the pipeline modules.
if platform.system() == "Windows":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .pipeline import main, process_building
