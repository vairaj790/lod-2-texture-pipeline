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
3. evaluates raw OSM obstruction before fitting, skips near-fully blocked
   candidates, extrudes every neighbouring footprint from its own validated
   DGM terrain elevation, and excludes rendered neighbouring buildings from
   SAM3 target association and weighted global-depth fitting;
4. reuses one SAM3 embedding to build both whole-model fit guidance and tighter
   target-wall visibility guidance, rejecting tree/wall/fence-obscured sources;
5. selects one native image with the best usable target visibility, or leaves
   the facade unresolved when every candidate is obstructed;
6. refits the selected candidate only when an OSM side exclusion changes its
   valid image evidence, while reusing that candidate's existing SAM3 masks;
7. intersects the fitted facade projection with the selected full-image
   building/occluder evidence, then propagates that mask through rectification
   and bounded Hough adjustment with nearest-neighbor transforms;
8. reuses the propagated mask for facade cleanup and LaMa filling, without a
   second SAM3 inference on the cropped or rectified wall;
9. applies an optional matching GeoTIFF to the roof; and
10. exports the textured model and diagnostic artifacts.

SAM3 guidance is anchored to the original projected building region. It helps
the fitter interpret building, roof, sky, vegetation, ground, known foreground
objects, and guarded generic foreground proposals without allowing an unrelated
building elsewhere in the image to capture the fit. The selected full-image
building and occluder masks remain in source-image coordinates and are reused
for downstream facade extraction.

OSM obstruction uses depth overlap in the candidate image as its authoritative
test. Nearby buildings closer than the target are rendered even when a narrow
ground-plan corridor would miss them; buildings elsewhere in the frame do not
count unless their rendered depth actually covers the projected target wall.

Global fitting is a bounded correction around the raw projection: scale,
translation, mean displacement, relative displacement, and projection overlap
all have hard acceptance limits. When SAM3 has no reliable target-wall support,
only a small micro-correction is permitted.

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
- the recommended Big-LaMa checkpoint or the lighter LaMa ONNX fallback
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

### 4. Download an inpainting model

The quality path uses the official native-resolution Big-LaMa generator on
PyTorch/CUDA. The fixed-512 OpenCV ONNX export remains a CPU-compatible
fallback. With `LAMA_BACKEND = "auto"`, the pipeline chooses Big-LaMa when its
prepared generator file and CUDA are available and otherwise uses ONNX. Set
`BIG_LAMA_DEVICE = "cpu"` explicitly if native Big-LaMa quality is worth the
substantially slower CPU runtime. Download both model variants if you want
`auto` mode to retain the ONNX fallback when CUDA or the native generator is
unavailable.

#### Recommended: native Big-LaMa

The preparation script verifies the official checkpoint SHA-256 before loading
it and writes a clean inference-only state file, avoiding LaMa's obsolete
Lightning/Hydra training environment.

PowerShell:

```powershell
New-Item -ItemType Directory -Force ..\lama_model | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" -OutFile "..\lama_model\big-lama.zip"
Expand-Archive -LiteralPath "..\lama_model\big-lama.zip" -DestinationPath "..\lama_model"
python prepare_big_lama_checkpoint.py
```

Bash:

```bash
mkdir -p ../lama_model
curl -L "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" -o ../lama_model/big-lama.zip
unzip ../lama_model/big-lama.zip -d ../lama_model
python prepare_big_lama_checkpoint.py
```

#### Fallback: fixed-512 ONNX LaMa

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
|   |-- big-lama/
|   |   |-- config.yaml
|   |   `-- models/
|   |       |-- best.ckpt
|   |       `-- generator.pt
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
    "LAMA_BACKEND": "auto",
    "BIG_LAMA_GENERATOR_PATH": "../lama_model/big-lama/models/generator.pt",
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

### Summarize a completed batch

Generate building- and wall-level CSV/JSON summaries from existing outputs:

```bash
python summarize_batch_outputs.py --outputs-dir outputs
```

Reports are written to `outputs/batch_statistics`. By default, final KMZ and
GLB files are also collected under `outputs/kmz_files` and `outputs/glb_files`;
pass `--no-copy` to generate statistics without copying the exports.

## Outputs

Each building receives its own directory under `OUTPUT_DIR`:

```text
outputs/
`-- building_48959353_3d/
    |-- building_48959353_3d__textured.glb
    |-- building_48959353_3d__textured.kmz
    |-- posttexture_base_repair.json
    |-- stage_timings.json
    |-- viewer_bundle.npz
    |-- viewer_index.json
    `-- wall_artifacts/
        |-- _global/
        `-- group_.../
```

Facade-group directories contain rectified textures, metadata, semantic and
fit diagnostics, masks, and an execution-ordered `debug_contact_sheet.png`.
Every wall-specific artifact is routed to its geometry group. A group for which
no source was selected contains only `group_summary.json` and the geometry
legend `debug_contact_sheet.png`.
Artifact creation is controlled by the `SAVE_*` settings.

Immediately before GLB/KMZ export, the finished model is checked for an uneven
base. An already-level model is left unchanged. Otherwise, the original
textured walls remain untouched, separate solid-colour wall skirts extend to
the building-wide minimum base elevation, and the old underside is replaced by
a flat base. Each skirt colour is estimated only from the corresponding wall's
UV-covered texture pixels; `posttexture_base_repair.json` records the decision,
elevations, added geometry, and sampled colours.

## Inpainting quality controls

The inpainting path extends valid facade RGB underneath the wall boundary before
inference, keeps the true hole mask separate from the dilated context mask, and
changes only masked pixels. This prevents boundary-connected holes from seeing
transparent black as valid context. Exported textures also receive a transparent
RGB gutter, while GLB facade materials use neutral color, non-metallic shading,
binary alpha, and clamped base-color sampling.

`LAMA_BACKEND = "auto"` is the recommended setting. Native Big-LaMa runs each
connected hole group with surrounding context and a configurable resolution
budget (`BIG_LAMA_CONTEXT_PX`, `BIG_LAMA_MAX_SIDE_PX`, and
`BIG_LAMA_MAX_PIXELS`). The ONNX fallback uses an aspect-preserving coarse pass
plus native-resolution high-frequency tiles; its overlap and detail blending are
controlled by the `LAMA_TILE_*` settings.

The official coarse-to-fine Big-LaMa feature refiner is available through
`BIG_LAMA_ENABLE_REFINEMENT`, but remains disabled by default. It retains
gradients through most of the generator and can take minutes or exceed 8 GB on
large walls. Use it as an offline experiment with
`BIG_LAMA_REFINEMENT_MAX_PIXELS`, not as the normal batch path.

Existing GLBs created before these changes must be regenerated or repatched to
receive the corrected material and sampler settings.

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
- Large one-sided holes remain underconstrained. An inpainting model can make
  them plausible, but cannot recover accurate windows, masonry, or objects that
  are absent from every source pixel; diffusion fallbacks would hallucinate
  those details rather than reconstruct them.
- Configuration is currently file-based; there is no packaged command-line
  interface.

## Licensing status

This repository does not yet declare a software license. No permission beyond
applicable default copyright law should be inferred until the project owner
adds a license. The redistribution rights and attribution requirements for
sample GeoJSON, GeoTIFF, Street View, OSM, DGM, SAM3, and LaMa materials must be
assessed separately from the source-code license.

`lod2_texture_pipeline/big_lama.py` adapts the feed-forward FFC generator and
feature-refinement method from
[`advimman/lama`](https://github.com/advimman/lama) and
[`geomagical/lama-with-refiner`](https://github.com/geomagical/lama-with-refiner),
which are published under Apache-2.0. Model weights are downloaded separately
and are not stored in this repository. See `THIRD_PARTY_NOTICES.md` and the
included upstream license copy under `LICENSES/`.

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
