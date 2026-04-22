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
  One GeoJSON per building. These are 3D wireframe-style building descriptions in 'EPSG:25832'.
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

The pipeline relies on three kinds of dependencies:

- Python packages for geospatial IO, geometry, deep learning, and mesh export
- model assets that are not installed from `pip`
- external API access for Google Street View

### Python Dependencies

Use either:

- `requirements.txt` for a `pip`-based setup, or
- `environment.yml` for a Conda setup

### External Assets And Services

You also need:

- a Google Street View Static API key with billing enabled
- the SAM3 package and its model weights
- the LaMa ONNX checkpoint at `lama_model/inpainting_lama_2025jan.onnx`

## Installation

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate lod2-texture-pipeline
```

### Option 2: venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

The repo keeps a code-first configuration style. Runtime paths and thresholds live in `lod2_texture_pipeline/config.py`.

Important fields:

- `GEOJSON_DIR`
- `GEOTIFF_DIR`
- `OUTPUT_DIR`
- `API_KEY`
- `LAMA_MODEL_PATH`
- `SAM3_PROMPT_FACADE`
- `SAM3_PROMPT_ROOF`

For public use, do not hardcode secrets into `config.py`. Instead, create a local untracked override file:

- `lod2_texture_pipeline/config_local.py`

This file is already ignored by Git and can override any uppercase config variable.

An example template is provided in `lod2_texture_pipeline/config_local.example.py`.

## Minimal Setup Checklist

Before running the pipeline, make sure:

1. Your GeoJSON files are in the folder pointed to by `GEOJSON_DIR`.
2. Your matching roof rasters are in the folder pointed to by `GEOTIFF_DIR`.
3. `API_KEY` is set through `config_local.py` or directly in `config.py`.
4. The LaMa checkpoint exists at `LAMA_MODEL_PATH`.
5. The `sam3` package and its required weights are installed and accessible.

## How To Run

Batch mode:

```bash
python run_batch.py
```

Single-building mode:

```bash
python single_test.py
```

The batch runner:

- scans `GEOJSON_DIR` for `*.geojson`
- searches for matching `.tif` or `.tiff` roof rasters
- loads SAM3 once
- processes each building sequentially
- exports one textured `glb` per building

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
- Paths are configured as absolute paths by default; for a public deployment you will usually want to override them locally.
- SAM3 weights are not bundled in this repository.
- The LaMa ONNX file is referenced by path and is not installed automatically.
- Google Street View requests depend on API quota, billing, and imagery availability.
- CPU execution is possible in principle, but SAM3 inference is likely to be much slower than GPU execution.

## Known Limitations

- There is no CLI yet; the workflow is controlled through `config.py` and optional local overrides.
- The repo assumes one roof GeoTIFF per building and filename-based matching.
- The segmentation prompts are fixed-text defaults and may need tuning for new regions or datasets.
- The current packaging documents the `sam3` dependency, but users may still need to align installation with the specific SAM3 distribution they use.

## Recommended Citation / Description For GitHub

If you want a short repo description for GitHub, this is a good summary:

`Pipeline for converting 3D GeoJSON building wireframes plus roof GeoTIFFs and Google Street View imagery into textured LoD2 GLB building models using SAM3 facade segmentation, orthorectification, and LaMa inpainting.`
