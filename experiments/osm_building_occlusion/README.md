# OSM External-Building Occlusion Diagnostic

This optional diagnostic reruns the production OSM-occlusion logic to inspect
whether a nearby **different building** blocks a target facade in an existing
Street View candidate. It does not replace or alter the main pipeline.

It does not search for new panoramas. It reads the exact `source_candidates`
already recorded in each production `*__ortho_meta.json` file, so candidate
discovery remains unchanged.

## Method

1. Download nearby OpenStreetMap building footprints through Overpass and cache
   the response locally.
2. Remove the OSM footprint corresponding to the LoD-2 target building.
3. Extrude the remaining footprints into simple 2.5D building prisms. Tagged
   `height` is preferred, then `building:levels`, then a configurable fallback.
4. Render the raw target-wall depth from every saved candidate camera, then run
   the same global-depth boundary fitter used by production. The exact accepted
   fit is reused for the selected source; alternative candidates are recomputed
   with the same fitting method.
5. Warp the target-wall depth into its corrected global-depth position. A
   target pixel is externally obstructed only when an OSM building is closer
   to the camera than the corrected target wall at that same image pixel.
6. Prefer candidates below the clear threshold, retaining the existing
   production ranking among clear candidates. If every candidate is obstructed,
   select the least-obstructed one and generate a white removal mask.

The white pixels in `02_selected_external_building_removal_mask.png` mean:
"exclude or inpaint this external-building content." The mask is restricted to
the **corrected global-depth wall projection**, not the original raw wall
projection. The transparent fallback preview is generated only when all
candidates are obstructed. In the contact sheet, transparent removed pixels are
shown with a gray checkerboard so that the removal remains visible after the
preview is placed on the opaque contact-sheet canvas.

Candidate overlays use these colors:

- cyan: original wall projection;
- magenta: corrected global-depth wall projection used by this experiment;
- orange: externally obstructed pixels inside the corrected wall.

The fitted target transform is not applied to the OSM blocker geometry. It is a
2D correction estimated specifically for the target model, rather than a new
physical camera pose, so each external building remains at its native projection
in the Street View image.

## Run

From the repository root:

```powershell
python -m experiments.osm_building_occlusion.run_experiment `
  --geojson sample_data/3d_geojsons/building_48959353_3d.geojson
```

Test one facade group:

```powershell
python -m experiments.osm_building_occlusion.run_experiment `
  --geojson sample_data/3d_geojsons/building_48959353_3d.geojson `
  --group g02
```

Results are written below
`experiments/osm_building_occlusion/results/<building>/`. Existing production
outputs are read-only inputs.

## Limitations

- OSM footprints and heights can be incomplete or stale.
- Buildings without height tags use the configured fallback height.
- Every blocker currently uses the target building's ground elevation. This is
  an approximation on sloped terrain.
- The fallback mask is in the selected native Street View image space and is
  already aligned to the corrected global-depth target wall. Production carries
  the source and mask through the same later crop and rectification transforms.
- OSM cannot identify trees, vehicles, scaffolding, or other non-building
  occluders; those remain the segmentation stage's responsibility.

Map data attribution: OpenStreetMap contributors, ODbL 1.0.
