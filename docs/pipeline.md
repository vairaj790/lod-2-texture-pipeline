# Pipeline stages

This document describes the default production path in execution order. Feature
flags can disable optional stages, but the coordinate and source-selection
invariants remain the same.

## Core invariants

- A facade group is processed from one selected native Street View image.
- Every candidate is evaluated independently before source selection.
- The original projected LoD2 model anchors semantic association and the fit
  search region.
- An OSM exclusion is a boolean mask in the selected image's existing
  coordinates. It does not create a new coordinate system or physically crop
  the canvas before the selected-source refit.
- Pre-fit SAM3 guidance and post-rectification SAM3 cleanup are separate stages
  with different purposes.
- The whole-model fit supplies one bounded image-space transform. It does not
  independently reshape model edges.

## 1. Load building geometry

`process_building` loads the 3D GeoJSON, resolves node IDs and typed edges, and
builds wall, roof, and base geometry. Explicit polygon surface records are used
when available; edge loops provide the fallback representation.

Wall fragments are grouped into line-compatible facade groups. A group may
contain one wall or several adjacent wall fragments that share one facade
texture.

## 2. Resolve camera elevation

When DGM camera elevation is enabled, the pipeline samples the official
Thuringia DGM1 data and first compares it with the model's base vertices.
Camera elevations use DGM only when enough base samples agree within the
configured tolerances.

If DGM is disabled, unavailable, outside its coverage, or inconsistent with the
model, the pipeline uses:

```text
camera elevation = model base elevation + FIXED_HEIGHT_M
```

DGM tiles are cached only in memory for the current run.

## 3. Discover Street View candidates

The pipeline searches a grid around the building footprint using the Google
Street View metadata endpoint. By default it requests outdoor imagery and
accepts Google-owned panoramas.

For each facade group, the geometric selector:

1. keeps cameras on the outward-facing side of the facade;
2. measures forward distance, lateral position, and frontality;
3. samples candidates along the facade; and
4. limits the result to `FACADE_GROUP_MAX_CANDIDATE_PANOS`.

If no sufficiently frontal candidate is found, an optional wider metadata
recovery search probes farther outward from the facade.

## 4. Fetch and project each candidate

Each candidate panorama is fetched separately with its own heading, pitch, and
field of view. The pipeline constructs the corresponding camera intrinsics and
pose, projects the target facade, and rejects invalid, behind-camera, or
line-like projections.

A complete-model z-buffer then estimates how much of the target wall is
self-occluded by nearer parts of the LoD2 model.

## 5. Build pre-fit SAM3 guidance for every candidate

Before candidate ranking, SAM3 computes an image embedding and evaluates a
fixed configured concept library. The default roles cover:

- target building and roof;
- sky and ground;
- vegetation such as trees and bushes; and
- foreground occluders such as vehicles.

This is automatic from the pipeline user's perspective; no interactive point,
box, or text prompt is requested for each image.

The raw rendered model mask defines a local search neighborhood. Building and
roof instances must overlap, or lie close to, that original projection before
they can become target evidence. This prevents a larger unrelated building
elsewhere in the frame from pulling the fit away from the projected model.

The guidance builder produces:

- target building and roof masks;
- roof, wall, base, and silhouette boundary evidence;
- a valid-evidence mask;
- excluded vegetation and occluder evidence; and
- metadata describing instance association and evidence support.

Image-frame borders and contour closures caused by viewport clipping are
excluded from fit evidence and are not drawn as model geometry.

## 6. Fit the projected whole model for every candidate

The global-depth fitter renders complete-model camera-forward depth and obtains
the visible projected roof, wall, and base edges. It searches for one bounded
similarity transform near the raw projection.

Default class priorities are:

| Model edge class | Weight |
| --- | ---: |
| Roof | 3.0 |
| Wall | 2.0 |
| Base | 0.35 |

Class contributions are normalized by visible length. A long base edge
therefore does not dominate solely because it provides more samples.

Semantic image boundaries and conventional image-edge evidence contribute only
where the valid-evidence mask permits them. Vegetation, vehicles, unsupported
areas, and image borders cannot score a candidate transform.

A transform is accepted only when it satisfies the configured evidence and
score-improvement checks. Otherwise the candidate retains its raw projection.

## 7. Score OSM obstruction

When enabled, the pipeline queries nearby OSM building footprints, removes
footprints belonging to the target building, estimates blocker heights, and
renders external blockers in the same candidate camera.

The obstruction fraction reduces the candidate's otherwise usable target-wall
visibility. It is not a simple clear-versus-blocked veto: a nearly complete wall
with a small obstruction may still be better than a clear image containing
only a small part of the target wall.

If OSM is unavailable, processing continues with the remaining geometric and
visibility criteria.

## 8. Select one native source

Candidate ranking first requires a valid, nondegenerate facade projection. The
ranking then considers:

- availability of target-model visibility evidence;
- net target visibility after frame loss, self-occlusion, and OSM obstruction;
- usable and self-visible target fractions;
- complete in-frame coverage;
- projected coverage and visible area; and
- camera distance as a late tie-breaker.

`FACADE_SOURCE_SELECTION_MODE = "auto"` preserves the wall-prism preference for
buildings whose facade groups are all single walls and uses projected-coverage
selection for genuine multi-wall groups.

Only the highest-ranked native image proceeds to texture extraction.

## 9. Refit the selected source only when needed

If the selected source has no OSM side exclusion, the pipeline reuses its
already evaluated candidate fit and SAM3 guidance.

If an external building obstructs the selected source, the obstruction-facing
divider is extended to an image-side exclusion. The pipeline then:

1. keeps the original image dimensions and coordinates;
2. reruns SAM3 guidance for the selected image;
3. removes the excluded side from valid fit evidence;
4. reruns the weighted whole-model global fit; and
5. passes the excluded region downstream as missing alpha for later filling.

This selected-source pass is a refit because the valid evidence changed. It is
not an unconditional second fit for every selected candidate.

## 10. Choose the downstream alignment

With `FACADE_ALIGNMENT_MODE = "depth_global"`, the accepted whole-model
transform supplies the authoritative wall projection used by cropping,
rectification, and texturing.

If the global fit is unavailable or rejected, the alignment selector keeps the
safe wall-only/raw fallback for that facade group.

## 11. Projection crop and orthorectification

The selected wall outline creates the source alpha mask. OSM-excluded pixels
inside that projection remain transparent.

A homography maps the source facade rectangle into metric wall-plane
coordinates. The output scale is controlled by `PIXELS_PER_METER`, with a
maximum pixel-count guard for large walls. Pixels outside the rectified wall
polygon are made transparent.

## 12. Bounded Hough adjustment

Hough line selection runs on the rectified RGB facade before the cleanup SAM3
stage. Search bands are restricted around expected left, right, and roof edges.

When accepted, the guided warp adjusts the rectified image within the configured
bounds. A single detected side may guide a progressive one-sided warp while the
missing side remains fixed.

## 13. Post-rectification SAM3 cleanup

SAM3 runs again on the Hough-adjusted, rectified facade. This stage removes
non-building content inside the already aligned wall projection.

The selected segmentation is clipped to the allowed wall/projection mask. It
cannot expand the wall or replace the global alignment. If segmentation support
is insufficient, the projection-defined texture remains unchanged.

An optional guarded rectified fit can apply a small final scale and translation,
but it is disabled by default.

## 14. Fill missing wall content

LaMa receives holes inside the wall polygon, including accepted cleanup
exclusions and OSM-derived transparent regions. Small components are removed,
the mask is dilated within the wall, and ONNX Runtime fills the remaining
pixels.

If LaMa is intentionally disabled, transparent/missing wall content is not
inpainted.

## 15. Texture roofs and export

When a matching GeoTIFF exists, its geotransform supplies roof UV coordinates
and the building footprint masks the roof texture. If the raster is absent or
cannot be used, the roof remains untextured.

The pipeline exports:

- a local-coordinate, Y-up textured `glb`;
- an optional textured `kmz`;
- per-wall metadata and viewer data; and
- grouped diagnostics and contact sheets.

See [Artifacts and contact sheets](artifacts.md) for output names and visual
legends.

## Safe fallbacks

| Condition | Result |
| --- | --- |
| No Street View candidates | Building processing stops without a facade result |
| Invalid or edge-on candidate projection | Candidate is rejected |
| Insufficient global-fit evidence | Raw or wall-only projection is retained |
| OSM or Overpass unavailable | Selection continues without external-blocker scoring |
| DGM unavailable or inconsistent | Model-base camera elevation fallback is used |
| Post-rectification SAM3 rejected | Projection-defined facade texture is retained |
| Matching GeoTIFF missing | Roof remains untextured |
