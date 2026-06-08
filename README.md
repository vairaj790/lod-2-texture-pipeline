# LoD2 Texture Pipeline

This repository builds textured LoD2 `glb` building models from:

- 3D GeoJSON building wireframes
- roof GeoTIFF imagery
- Google Street View facade imagery

The pipeline reconstructs each building as wall, roof, and base meshes, finds suitable Street View panoramas for facade walls, segments facades with SAM3, rectifies facade imagery into wall-plane textures, fills missing wall texture regions with LaMa, maps roof imagery from GeoTIFFs, and exports a textured LoD2 `glb`.

## What The Pipeline Does

At a high level, each building goes through these stages:

1. Load a 3D GeoJSON building skeleton.
2. Search Google Street View around the building footprint.
3. Select suitable panoramas for visible walls.
4. Fetch Street View facade images.
5. Segment facades with SAM3.
6. Orthorectify facade imagery into wall coordinates.
7. Fill missing wall texture regions with LaMa ONNX inpainting.
8. Texture roofs from GeoTIFF imagery.
9. Export the final textured LoD2 `glb`.

## Inputs

The pipeline expects two synchronized input collections:

- `3d_geojsons/*.geojson`  
  One GeoJSON per building. The expected CRS is `EPSG:25832`.

- `geotiffs/*.tif` or `*.tiff`  
  One roof raster per building. Filenames are matched against the GeoJSON basename, with `_3d` stripped from the GeoJSON stem when needed.

Sample inputs are included in:

- `sample_data/3d_geojsons/`
- `sample_data/geotiffs/`

## Outputs

For each input building, the pipeline creates a folder inside `OUTPUT_DIR` containing:

- rectified wall textures as PNGs
- segmentation and rectification debug overlays
- optional LaMa hole masks
- per-wall metadata JSON files
- `viewer_index.json`
- `viewer_bundle.npz`
- the final textured `glb`

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/vairaj790/lod-2-texture-pipeline.git
cd lod-2-texture-pipeline
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate lod2_texture_pipeline
```

### 3. Install PyTorch

This is the tested PyTorch setup used for the current pipeline:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify it:

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torch.version.cuda); print(torchvision.__version__); print(torch.cuda.is_available())"
```

### 4. Install SAM3

SAM3 is installed separately from the official repository. Use the pinned commit below:

```bash
cd ..
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 11dec2936de97f2857c1f76b66d982d5a001155d
pip install .
cd ../lod-2-texture-pipeline
```

Verify SAM3:

```bash
python -c "import sam3; print('sam3 ok')"
```

### 5. Download the LaMa ONNX model

LaMa is used as a separate model asset, similar to SAM3. Keep it outside the repository folder.

From inside `lod-2-texture-pipeline`, run:

```bash
cd ..
mkdir -p lama_model
cd lama_model
wget -O inpainting_lama_2025jan.onnx "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx?download=true"
cd ../lod-2-texture-pipeline
```

After this, the expected structure is:

```text
parent_folder/
├── lod-2-texture-pipeline/
├── sam3/
└── lama_model/
    └── inpainting_lama_2025jan.onnx
```

### 6. Configure local paths and API key

Do not put private paths or API keys directly into `config.py`.

Create a local override file:

```bash
cp lod2_texture_pipeline/config_local.example.py lod2_texture_pipeline/config_local.py
```

Edit:

```text
lod2_texture_pipeline/config_local.py
```

Typical local overrides are:

```python
LOCAL_CONFIG = {
    "GEOJSON_DIR": "sample_data/3d_geojsons",
    "GEOTIFF_DIR": "sample_data/geotiffs",
    "OUTPUT_DIR": "outputs",
    "API_KEY": "YOUR_GOOGLE_STREET_VIEW_API_KEY",
    "LAMA_MODEL_PATH": "../lama_model/inpainting_lama_2025jan.onnx",
}
```

`config_local.py` is ignored by Git.

## Linux Shared-Library Note

On some Linux systems, compiled packages may accidentally load older system libraries instead of Conda libraries. If you see an error similar to:

```text
GLIBCXX_3.4.29 not found
```

run:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Then retry the command.

## Verify The Installation

After installing the environment, PyTorch, SAM3, and placing the LaMa ONNX file, run:

```bash
python -c "import numpy, rasterio, geopandas, torch, torchvision, sam3, onnxruntime, lod2_texture_pipeline; print('repo import ok')"
```

## How To Run

### Batch mode

Batch mode uses folder paths from `config.py` and optional overrides from `config_local.py`.

```bash
python run_batch.py
```

The batch runner:

- scans `GEOJSON_DIR` for `*.geojson`
- searches for matching `.tif` or `.tiff` roof rasters
- loads SAM3 once
- processes each building sequentially
- exports one textured `glb` per building

### Single-building mode

`single_test.py` is for testing one building only.

It uses one explicit GeoJSON file and one explicit GeoTIFF file. Edit these paths inside `single_test.py` before running:

```python
GEOJSON_PATH = REPO_ROOT / "sample_data" / "3d_geojsons" / "building_48959353_3d.geojson"
GEOTIFF_PATH = REPO_ROOT / "sample_data" / "geotiffs" / "building_48959353.tif"
```

Then run:

```bash
python single_test.py
```

## Configuration Files

Main public config:

```text
lod2_texture_pipeline/config.py
```

Local private override:

```text
lod2_texture_pipeline/config_local.py
```

Example local config:

```text
lod2_texture_pipeline/config_local.example.py
```

Use `config.py` for public-safe defaults and `config_local.py` for machine-specific paths, API keys, and private settings.

## Expected GeoJSON Structure

The loader expects features with properties similar to:

- `type`: one of `roof`, `base`, `wall`, `wall_center`
- `source`
- `target`
- optionally `component_id`
- optionally `loop_id`
- optionally `ring_order`

The sample files in `sample_data/3d_geojsons/` show the intended structure.

## Notes

- The pipeline currently assumes CRS `EPSG:25832` and converts to `EPSG:4326` for Street View queries.
- Google Street View requests require a valid API key.
- SAM3 weights are downloaded/managed by SAM3 and are not bundled in this repository.
- The LaMa ONNX model is downloaded separately into a sibling `lama_model/` folder.
- Full SAM3 inference is intended for GPU execution. CPU execution may be very slow.
- There is no CLI yet; the workflow is controlled through `config.py`, `config_local.py`, `run_batch.py`, and `single_test.py`.
