# LoD-2 Texture Pipeline

This repository is a refactored GitHub-ready version of the original monolithic LoD-2 texture pipeline.

## Repository layout

- `run_batch.py` – top-level entry point for batch execution.
- `lod2_texture_pipeline/config.py` – all embedded paths, constants, and thresholds.
- `lod2_texture_pipeline/geojson_io.py` – GeoJSON loading and loop grouping.
- `lod2_texture_pipeline/streetview.py` – Street View pano discovery and geometric pano selection.
- `lod2_texture_pipeline/projection.py` – camera math, projection, SAM3 loading, rectification, and coverage tiling.
- `lod2_texture_pipeline/quadfit.py` – quad fitting, Hough line detection, affine wall fitting.
- `lod2_texture_pipeline/inpainting.py` – LaMa ONNX hole filling.
- `lod2_texture_pipeline/mesh.py` – wall/roof/base meshing and roof masking helpers.
- `lod2_texture_pipeline/utils.py` – shared helpers, naming, overlays, and bundle export.
- `lod2_texture_pipeline/pipeline.py` – per-building processing and batch driver.

## How to run

1. Open `lod2_texture_pipeline/config.py`.
2. Adjust the folder paths and any constants you need.
3. Activate your conda environment.
4. Run:

```python
python run_batch.py
```

## Notes

- This repository keeps the original no-CLI workflow: paths and parameters stay embedded in `config.py`.
- The SAM3 package and its weights still need to be available in your environment.
- The Google API key is still embedded exactly as in the source script.
