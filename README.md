# LoD2 Texture Pipeline

This repository textures LoD2 building geometry with Google Street View facade
imagery and georeferenced roof imagery. It uses SAM3-guided whole-model
alignment, OpenStreetMap building-occlusion checks, bounded Hough adjustment,
and LaMa inpainting, then exports textured `glb` and `kmz` models.

The central source-selection rule is simple: each facade group is textured from
exactly one native Street View image. Candidate images are evaluated
independently and are never combined into a new processing image.

## Pipeline overview

For each building, the default pipeline:

1. loads the 3D GeoJSON and constructs wall, roof, and base meshes;
2. discovers nearby outdoor, Google-owned Street View panoramas;
3. evaluates every candidate using whole-model visibility, SAM3 semantic
   guidance, weighted global-depth fitting, and OSM obstruction;
4. selects one native image with the best usable target visibility;
5. reruns the selected fit only when an OSM side exclusion changes its valid
   image evidence;
6. crops the facade projection, rectifies it into wall coordinates, and applies
   bounded Hough adjustment;
7. runs a separate post-rectification SAM3 cleanup and fills missing wall
   content with LaMa;
8. applies an optional matching GeoTIFF to the roof; and
9. exports the textured model and diagnostic artifacts.

SAM3 guidance is anchored to the original projected building region. It helps
the fitter interpret building, roof, sky, vegetation, ground, and vehicle
evidence without allowing an unrelated building elsewhere in the image to
capture the fit.

See [Pipeline stages](docs/pipeline.md) for the detailed execution order and
[Artifacts and contact sheets](docs/artifacts.md) for the output glossary.

## Inputs

The batch runner reads:

- `GEOJSON_DIR/*.geojson`: one 3D building file per model;
- `GEOTIFF_DIR/*.tif` or `*.tiff`: optional georeferenced roof imagery.

GeoJSON names ending in `_3d` are matched to roof rasters after removing that
suffix. For example:

```text
sample_data/
|-- 3d_geojsons/
|   `-- building_48959353_3d.geojson
`-- geotiffs/
    `-- building_48959353.tif
```

The GeoJSON line features use `type` values such as `wall`, `roof`, `base`, and
optionally `wall_center`. Line features require `source` and `target`
properties. `component_id`, `loop_id`, `ring_order`, and explicit polygon
surface records are supported when present.

The current geometry and Street View transforms assume `EPSG:25832`. A missing
or unusable GeoTIFF does not stop facade processing; the affected roof remains
untextured.

## Requirements

- Git and Conda
- Python 3.12, as specified by `environment.yml`
- a Google Cloud API key with the Street View Static API enabled
- access to the gated Hugging Face model `facebook/sam3`
- the LaMa ONNX model described below
- internet access for Street View, SAM3 weights, and the optional OSM and DGM
  services

An NVIDIA GPU is strongly recommended for SAM3. The tested installation uses
PyTorch 2.10.0, torchvision 0.25.0, and CUDA 12.8. The code can fall back to
CPU, but SAM3 inference can be very slow.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vairaj790/lod-2-texture-pipeline.git
cd lod-2-texture-pipeline
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate lod2_texture_pipeline
```

Install the tested CUDA build of PyTorch:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install the pinned SAM3 revision

Install SAM3 as a sibling of this repository:

```bash
cd ..
git clone https://github.com/facebookresearch/sam3.git
git -C sam3 checkout 11dec2936de97f2857c1f76b66d982d5a001155d
python -m pip install ./sam3
cd lod-2-texture-pipeline
```

Request access to [`facebook/sam3`](https://huggingface.co/facebook/sam3),
accept its terms, and authenticate locally:

```bash
hf auth login
```

SAM3 downloads its weights through Hugging Face when the pipeline first loads
the model.

### 4. Download the LaMa ONNX model

The default configuration expects the model in a sibling `lama_model`
directory.

PowerShell:

```powershell
New-Item -ItemType Directory -Force ..\lama_model | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx?download=true" -OutFile "..\lama_model\inpainting_lama_2025jan.onnx"
```

Bash:

```bash
mkdir -p ../lama_model
curl -L "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx?download=true" -o ../lama_model/inpainting_lama_2025jan.onnx
```

The resulting layout is:

```text
parent_directory/
|-- lama_model/
|   `-- inpainting_lama_2025jan.onnx
|-- lod-2-texture-pipeline/
`-- sam3/
```

### 5. Create the local configuration

Do not place API keys or machine-specific paths in the tracked
`lod2_texture_pipeline/config.py`.

PowerShell:

```powershell
Copy-Item lod2_texture_pipeline\config_local.example.py lod2_texture_pipeline\config_local.py
```

Bash:

```bash
cp lod2_texture_pipeline/config_local.example.py lod2_texture_pipeline/config_local.py
```

Edit `lod2_texture_pipeline/config_local.py`. A minimal sample configuration is:

```python
LOCAL_CONFIG = {
    "GEOJSON_DIR": "sample_data/3d_geojsons",
    "GEOTIFF_DIR": "sample_data/geotiffs",
    "OUTPUT_DIR": "outputs",
    "API_KEY": "YOUR_GOOGLE_STREET_VIEW_API_KEY",
    "LAMA_MODEL_PATH": "../lama_model/inpainting_lama_2025jan.onnx",
}
```

Relative paths are interpreted from the directory in which the runner is
started. Run the commands below from the repository root, or use absolute paths.
`config_local.py` is ignored by Git.

## Verify the installation

Import the main runtime dependencies:

```bash
python -c "from PIL import Image; import torch, torchvision, cv2, rasterio, geopandas, sam3, onnxruntime, lod2_texture_pipeline; print('repository import OK'); print('CUDA available:', torch.cuda.is_available())"
```

Run the unit tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Run the pipeline

### Included single-building sample

`single_test.py` uses the committed `building_48959353` GeoJSON and GeoTIFF:

```bash
python single_test.py
```

By default, the API key, output directory, LaMa path, and optional feature
overrides come from `config_local.py`. Run `python single_test.py --help` to
override the input files or output directory, or to run without a roof
GeoTIFF.

### Batch processing

Place GeoJSON files and optional matching GeoTIFFs in the configured input
directories, then run:

```bash
python run_batch.py
```

The batch runner loads SAM3 once, processes the GeoJSON files sequentially, and
continues to the next building if one building fails.

## Outputs

Each building receives its own directory under `OUTPUT_DIR`:

```text
outputs/
`-- building_48959353_3d/
    |-- building_48959353_3d__textured.glb
    |-- building_48959353_3d__textured.kmz
    |-- stage_timings.json
    |-- viewer_bundle.npz
    |-- viewer_index.json
    `-- wall_artifacts/
        |-- _global/
        |-- group_.../
        `-- _unassigned/
```

Facade-group directories contain rectified textures, metadata, semantic and
fit diagnostics, masks, and an execution-ordered `debug_contact_sheet.png`.
Artifact creation is controlled by the `SAVE_*` settings.

## API data, privacy, and caches

Street View metadata and image requests are sent to Google and may incur API
charges. Review the [Street View Static API
policies](https://developers.google.com/maps/documentation/streetview/policies)
before processing or sharing imagery, and never commit an API key.

Persistent Street View caching is disabled by default. Each normal run fetches
the required Street View responses without retaining a reusable on-disk image
cache. If you explicitly enable `ENABLE_STREETVIEW_CACHE`, you are responsible
for securing, expiring, and using that cache in accordance with the provider's
terms.

Diagnostic overlays, contact sheets, and selected-source artifacts can still
contain Street View pixels; they are outputs rather than reusable request
caches. Apply the same access, retention, and sharing care to those files.

OSM Overpass responses use a separate cache under
`cache/osm_building_occlusion` by default. Thuringia DGM tiles are retained only
in memory during a run. Generated outputs, local configuration, model weights,
and caches should remain outside version control.

## Current limitations

- The source CRS and geographic transforms currently assume `EPSG:25832`.
- Automatic DGM camera elevation uses the official Thuringia DGM1 service and
  is region-specific. If validation or sampling fails, camera height falls back
  to the model base elevation plus the configured fixed height.
- Street View quality and coverage depend on Google imagery availability.
- OSM obstruction reasoning depends on the completeness and height attributes
  of nearby OSM building footprints.
- SAM3 model access is gated, and practical runtime normally requires a
  CUDA-capable GPU.
- Configuration is currently file-based; there is no packaged command-line
  interface.

## Licensing status

This repository does not yet declare a software license. No permission beyond
applicable default copyright law should be inferred until the project owner
adds a license. The redistribution rights and attribution requirements for
sample GeoJSON, GeoTIFF, Street View, OSM, DGM, SAM3, and LaMa materials must be
assessed separately from the source-code license.

## Troubleshooting

On Linux, a compiled dependency may report a missing `GLIBCXX` symbol. Install
the Conda runtime libraries:

```bash
conda install -c conda-forge libstdcxx-ng libgcc-ng
```

If the process still loads an older system library, retry from the activated
environment with:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

If the LaMa file is intentionally unavailable, set `ENABLE_LAMA_FILL = False`
in `config_local.py`. If SAM3 cannot download its checkpoint, confirm that
model access was approved and repeat `hf auth login`.

## Documentation

- [Detailed pipeline stages](docs/pipeline.md)
- [Artifacts and contact sheets](docs/artifacts.md)
