# Artifacts and contact sheets

Each processed building receives a directory under `OUTPUT_DIR`. With the
default diagnostic settings, the root contains final deliverables and compact
machine-readable summaries, while detailed images and metadata are organized by
facade group under `wall_artifacts`.

## Directory layout

```text
outputs/
`-- <geojson_stem>/
    |-- <geojson_stem>__textured.glb
    |-- <geojson_stem>__textured.kmz
    |-- stage_timings.json
    |-- viewer_bundle.npz
    |-- viewer_index.json
    `-- wall_artifacts/
        |-- _global/
        |   |-- <geojson_stem>__debug_facade_groups.json
        |   |-- <geojson_stem>__debug_facade_groups_topdown.png
        |   |-- <geojson_stem>__debug_facade_groups_unwrapped.png
        |   `-- wall_group_image_projections/
        |-- group_<id>__<tag>__walls_<range>/
        |   |-- debug_contact_sheet.png
        |   |-- group_summary.json
        |   `-- stage artifacts...
        `-- _unassigned/
```

Some files are conditional on feature and `SAVE_*` settings. Missing stage
numbers or artifacts do not necessarily indicate an error.

## Root deliverables

| Artifact | Meaning |
| --- | --- |
| `*__textured.glb` | Final textured LoD2 scene. The default export uses local glTF Y-up coordinates and records the EPSG:25832 origin in asset metadata. |
| `*__textured.kmz` | Google Earth-compatible textured model when `EXPORT_KMZ` is enabled. |
| `viewer_index.json` | Per-wall texture, camera, source-selection, fit, and diagnostic metadata. |
| `viewer_bundle.npz` | Compact NumPy bundle containing model corners, typed edges, wall quads, wall metadata, and a serialized viewer index. |
| `stage_timings.json` | Wall-clock timing summary for building-level and facade-stage operations. |

## Facade-group folders

Each `group_*` folder represents the wall fragments listed in its
`group_summary.json`. Its contact sheet begins with a fixed-orientation
untextured model panel. Highlighted red faces identify the wall fragments
represented by that folder; the blue point is the selected Street View camera
when one is available.

The model panel is an index view for orientation. It is not rendered from the
Street View camera.

## Contact-sheet order

`debug_contact_sheet.png` sorts cards by execution stage rather than filename.
Cards are included only when their corresponding stage ran and saved a visual.

| Stage label | What it shows |
| --- | --- |
| `00 untextured model + highlighted group` | The complete model with this facade group's wall fragments highlighted. |
| `01 candidate NN: SAM3-guided whole-model global fit` | One combined card per candidate: semantic guidance, raw and accepted fitted whole-model geometry, OSM obstruction, fit status, and score gain. A green check marks the selected candidate. |
| `02 selected native processing image` | The one image selected for downstream processing. This standalone file is normally omitted from the contact sheet because the selected stage-01 card already identifies it. |
| `03a selected source: raw depth before selected fit/refit` | Whole-model depth rendered from the selected camera before the selected-source fit is applied. |
| `03b OSM exclusion mask` | White pixels are excluded from selected-source refit evidence. Present only when an OSM side exclusion is required. |
| `03c OSM preview only` | Checkerboard visualizes excluded source pixels. The image canvas and coordinates have not been cropped. |
| `03d full-image SAM3 guidance before selected fit/refit` | Semantic target, boundary, search-region, and occluder guidance for the selected-source pass. |
| `03e ACTUAL refit evidence` | Exact valid image evidence used by the selected-source fit/refit. Checkerboard pixels do not contribute to the objective. |
| `03f raw + fitted whole-model projection` | Whole-model geometry before and after the accepted selected fit/refit, using the `03e` evidence. |
| `03g raw silhouette + dashed fitted boundary shift` | Compact shape-only view of the global transform. |
| `04 processing wall projection` | Selected wall projection used by downstream extraction, or a comparison view when enabled. |
| `05 side crop / LR band` | Side-band extraction and protection region around the selected projection. |
| `06 facade cropped by selected wall projection` | RGBA facade content retained for rectification. |
| `07 projection-cropped facade after rectification` | Rectified facade before bounded Hough adjustment. |
| `08a bounded Hough line selection before SAM` | Accepted or rejected candidate line evidence inside bounded search regions. |
| `08b bounded Hough warp adjustment before SAM` | Rectified image after an accepted Hough-guided warp. |
| `09a post-rectification cleanup SAM instances after Hough` | Candidate SAM3 facade masks on the Hough-adjusted rectified image. |
| `09b post-rectification cleanup SAM + guarded edge adjustment` | Selected cleanup mask and guarded result. |
| `10 guarded rectified edge adjustment` | Optional final bounded scale/translation, normally absent because it is disabled by default. |
| `11 LaMa hole mask` | White pixels filled by LaMa inside the wall region. |
| `12 final rectified texture` | RGBA texture assigned to the facade group. |
| `13 final texture overlay` | Final texture with the model wall boundary overlaid. |

The contact sheet deliberately omits some redundant full-resolution forensic
files. This keeps candidate guidance and candidate fitting together in a single
stage-01 card and avoids implying that two separate candidate fits occurred.

## Combined candidate-card legend

The stage-01 card is the main source-selection diagnostic.

| Visual | Meaning |
| --- | --- |
| Translucent cyan fill | SAM3 target-building region associated with the raw model projection |
| Translucent yellow fill | SAM3 target-roof region |
| Translucent pink fill | Tree, vegetation, vehicle, or other evidence excluded from fitting |
| Yellow guide | Roof boundary evidence, weighted `3.0` |
| Green guide | Wall boundary evidence, weighted `2.0` |
| Gray guide | Base boundary evidence, weighted `0.35` |
| Solid cyan model edges | Raw projected visible whole-model geometry |
| Dashed magenta model edges | Accepted fitted whole-model geometry |
| Orange outline | External OSM building obstruction |
| Darkened image area | Outside the projection-local semantic search region |
| Green check badge | Candidate selected for downstream processing |

The colored SAM3 guides are image-derived fitting evidence. They are not model
edges. The solid cyan and dashed magenta lines are the actual raw and fitted
whole-model projections.

## Checkerboard convention

Checkerboard pixels represent transparent or ignored evidence in a preview.
They do not indicate that the image coordinate system was cropped or shifted.

In particular:

- the `03c` OSM preview remains the complete selected image;
- the `03e` evidence preview shows pixels that cannot score the refit; and
- RGB images, semantic masks, depth, projected geometry, and exclusion masks
  remain aligned in the same selected-image coordinate system.

## Depth artifacts

For a prefix ending in `__model_depth`, the pipeline may save:

| Artifact | Meaning |
| --- | --- |
| `*__model_depth.npy` | Camera-forward depth in metres as `float32`, with `NaN` outside the rendered model. |
| `*__model_depth_mm_u16.png` | Millimetre depth as unsigned 16-bit PNG, with zero outside the model. |
| `*__model_depth_visual.png` | Colorized preview for inspection; not a metric input. |

The metric array and 16-bit PNG are generally kept as full-resolution forensic
artifacts rather than contact-sheet cards.

## Semantic and fit artifacts

Common full-resolution files include:

| Name pattern | Meaning |
| --- | --- |
| `*__source_panoNN_overlay.png` | Combined per-candidate semantic, global-fit, and OSM diagnostic used in stage 01. |
| `*__source_panoNN__prefit_semantic_guidance.png` | Standalone per-candidate SAM3 guidance. Saved for forensics but omitted from the contact sheet. |
| `*__model_depth_prefit_semantic_guidance.png` | Selected-source semantic guidance before its fit or OSM-masked refit. |
| `*__selected_depth_global_fit_evidence_preview.png` | Exact selected-source evidence support. |
| `*__whole_model_depth_boundary_fit.png` | Raw and fitted whole-model geometry in the selected image. |
| `*__model_depth_silhouette_mask.png` | Raw model silhouette with the fitted boundary shift. |
| `*__model_depth_boundary_fit_meta.json` | Transform, score, evidence, semantic-boundary, and acceptance metadata. |
| `*__projection_cropped_facade.png` | Source facade retained by the selected projection. |

Viewport-border contour closures are filtered before fitting and drawing. Lines
that merely follow the image boundary are therefore not treated as building
geometry.

## Rectification and texture artifacts

| Name pattern | Meaning |
| --- | --- |
| `*__ortho_prefit_overlay.png` | Rectified projection before Hough adjustment. |
| `*__hough_overlay.png` | Bounded Hough line candidates and selected lines. |
| `*__hough_warp_overlay.png` | Result of the Hough-guided warp. |
| `*__post_rectification_sam3_instances_overlay.png` | SAM3 instances evaluated after Hough adjustment. |
| `*__post_rectification_sam3_overlay.png` | Selected bounded cleanup result. |
| `*__ortho_lama_mask.png` | LaMa fill mask. |
| `*__ortho.png` | Final rectified facade texture. |
| `*__ortho_overlay.png` | Final facade texture plus wall boundary. |
| `*__ortho_meta.json` | Homographies, dimensions, source metadata, alignment selection, Hough result, SAM3 cleanup, and LaMa information. |

## Global diagnostics

The `_global` directory contains building-wide facade-group views:

- `*__debug_facade_groups_topdown.png` shows grouping in plan view;
- `*__debug_facade_groups_unwrapped.png` shows facade groups in unwrapped wall
  coordinates;
- `*__debug_facade_groups.json` records group membership and geometry; and
- `wall_group_image_projections/` stores projection views used to inspect
  candidate geometry across facade groups.

## Reading acceptance states

Most fitting metadata and overlays distinguish three outcomes:

- `accepted`: the bounded transform passed support and improvement checks;
- `raw fallback`: the fit was evaluated but the raw projection was safer; and
- `unavailable` or `failed`: required evidence could not be produced.

A fallback is intentional behavior, not necessarily a pipeline failure. The
downstream alignment metadata in `viewer_index.json` and `*__ortho_meta.json`
records which projection was authoritative.
