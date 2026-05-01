@'
# LoD2 Texture Pipeline

This repository builds textured LoD2 `glb` building models from:

- 3D GeoJSON building wireframes
- roof GeoTIFF imagery
- Google Street View facade imagery

The pipeline reconstructs each building as a set of wall, roof, and base meshes, finds suitable Street View panoramas for every wall, segments facades with SAM3, rectifies the selected facade imagery into wall-plane textures, optionally fills missing texture regions with LaMa, maps roof imagery from GeoTIFFs, and exports a textured LoD2 `glb`.

## What The Pipeline Does

At a high level, each building goes through these stages:

1. Load a 3D GeoJSON building skeleton.  
   The GeoJSON is expected to contain `roof`, `base`, `wall`, and optional `wall_center` features. Base, roof, and wall loops are reconstructed from the edge graph.

2. Search Google Street View around the footprint.  
   A buffered search grid is generated around the building base, candidate panoramas are collected with the Street View metadata API, and duplicate or out-of-zone panos are filtered out.

3. Select one best pano per wall.  
   For each wall face, the code computes an outward-facing prism in front of the wall and chooses the most suitable panorama geometrically.

4. Fetch facade imagery and ensure full wall coverage.  
   The wall quad is projected into the selected Street View image. If one request does not cover the wall well enough, the pipeline widens the view or fetches multiple yaw/pitch tiles and stitches them into a mosaic.

5. Segment facades with SAM3.  
   SAM3 is run with separate prompts for facade and roof. The pipeline keeps the best facade instance, removes roof pixels, cleans the mask, and saves debugging overlays.

6. Orthorectify the facade into wall coordinates.  
   The segmented wall is warped from perspective image space into a metric wall-plane texture. An additional quadrilateral fit and optional Hough-guided refinement help align the texture with the expected wall geometry.

7. Fill missing wall texture with LaMa.  
   Transparent holes inside the rectified wall polygon can be filled with a LaMa ONNX model loaded through OpenCV DNN.

8. Texture roofs from GeoTIFFs.  
   Roof meshes are triangulated from the roof edges, UVs are derived from GeoTIFF coordinates, and masking is applied so each roof island receives only its own roof pixels.

9. Export a textured LoD2 `glb`.  
   The final scene is written as a `glb`, together with per-wall texture artifacts, debug overlays, and viewer metadata bundles.

## Inputs

The pipeline expects two synchronized input collections:

- `3d_geojsons/*.geojson`  
  One GeoJSON per building. These are 3D wireframe-style building descriptions in `EPSG:25832`.
- `geotiffs/*.tif` or `*.tiff`  
  One roof raster per building. Filenames are matched against the GeoJSON basename, with `_3d` stripped from the GeoJSON stem when needed.

Sample inputs are included in:

- `sample_data/3d_geojsons/`
- `sample_data/geotiffs/`

## Outputs

For each input building, the pipeline creates a folder inside `OUTPUT_DIR` containing:

- rectified wall textures as PNGs
- per-wall JSON metadata describing camera pose and rectification transforms
- segmentation and rectification debug overlays
- optional LaMa hole masks
- `viewer_index.json`
- `viewer_bundle.npz`
- the final textured `glb`

## Requirements

The full intended workflow uses:

- Python 3.12
- the Conda environment from `environment.yml`
- a separate PyTorch installation
- a separate SAM3 installation
- a Google Street View Static API key
- the LaMa ONNX checkpoint at `lama_model/inpainting_lama_2025jan.onnx`

The repository no longer uses `requirements.txt` for the main setup.

## Installation

### 1. Clone this repository

    git clone https://github.com/vairaj790/lod-2-texture-pipeline.git
    cd lod-2-texture-pipeline

### 2. Create the base environment

    conda env create -f environment.yml
    conda activate lod2_texture_pipeline

### 3. Install PyTorch separately

    pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

### 4. Install SAM3 separately

Clone SAM3 next to this repository folder and install it in editable mode:

    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
    pip install -e .
    cd ..

### 5. Install additional packages required for the full SAM3 import path

    pip install "setuptools<81" einops pycocotools psutil

On Windows only:

    pip install "triton-windows<3.7"

### 6. Verify the installation

    python -c "import numpy, rasterio, torch, torchvision, sam3, lod2_texture_pipeline; print('repo import ok')"

## Configuration

The repo uses a code-first configuration style.

### Main config

Runtime defaults and pipeline parameters live in:

- `lod2_texture_pipeline/config.py`

This file should remain public-safe and generic.

### Local private override

Machine-specific paths and secrets should go in:

- `lod2_texture_pipeline/config_local.py`

This file is ignored by Git and can override any uppercase variable from `config.py`.

An example template is provided in:

- `lod2_texture_pipeline/config_local.example.py`

Typical local overrides include:

- `GEOJSON_DIR`
- `GEOTIFF_DIR`
- `OUTPUT_DIR`
- `API_KEY`
- `LAMA_MODEL_PATH`

## Minimal Setup Checklist

Before running the pipeline, make sure:

1. Your Google Street View API key is set in `config_local.py`.
2. The LaMa checkpoint exists at `LAMA_MODEL_PATH`.
3. SAM3 is installed and importable.
4. Your batch input folders are set correctly in `config.py` or `config_local.py`.
5. Your sample or test files actually exist at the paths you are using.

## How To Run

### Batch mode

Batch mode uses the folder-based settings from `config.py` and optional overrides from `config_local.py`.

    python run_batch.py

The batch runner:

- scans `GEOJSON_DIR` for `*.geojson`
- searches for matching `.tif` or `.tiff` roof rasters
- loads SAM3 once
- processes each building sequentially
- exports one textured `glb` per building

### Single-building mode

`single_test.py` is for testing one building only.

For this script, edit the single input file paths directly inside `single_test.py` before running it.

    python single_test.py

## Expected GeoJSON Structure

The loader expects features with properties similar to:

- `type`: one of `roof`, `base`, `wall`, `wall_center`
- `source`
- `target`
- optionally `component_id`
- optionally `loop_id`
- optionally `ring_order`

The sample files in `sample_data/3d_geojsons/` show the intended structure.

## Reproducibility Notes

- The pipeline currently assumes CRS `EPSG:25832` and converts to `EPSG:4326` for Street View queries.
- SAM3 weights are not bundled in this repository.
- The LaMa ONNX file is referenced by path and is not installed automatically.
- Google Street View requests depend on API access.
- CPU execution is possible in principle, but SAM3 inference is likely to be much slower than GPU execution.

## Known Limitations

- There is no CLI yet; the workflow is controlled through `config.py`, `config_local.py`, and `single_test.py`.
- The repo assumes one roof GeoTIFF per building and filename-based matching.
- The segmentation prompts are fixed-text defaults and may need tuning for new regions or datasets.
- SAM3 is installed separately and may require platform-specific dependency handling.
'@ | Set-Content README.md