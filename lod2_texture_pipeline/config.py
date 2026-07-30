# -*- coding: utf-8 -*-
"""Central configuration for the LoD-2 texture pipeline."""

import pyproj

# ======================= USER CONFIG (BATCH) =======================
GEOJSON_DIR = r"sample_data/3d_geojsons"
GEOTIFF_DIR = r"sample_data/geotiffs"
OUTPUT_DIR  = r"outputs"

API_KEY      = ""
SAM3_PROMPT_FACADE = "building facade"
SAM3_PROMPT_FACADE_REFINEMENT = "building walls"
SAM3_PROMPT_ROOF   = "roof"
ROOF_SUBTRACT_DILATE_PX = 2
ENABLE_FACADE_PROMPT_REFINEMENT = True
FACADE_REFINEMENT_MIN_WALL_PIXELS = 500
FACADE_REFINEMENT_MIN_PRIMARY_WALL_RATIO = 0.18

SV_SIZE          = "640x640"
FOV_MIN, FOV_MAX = 15.0, 120.0
FOV_MARGIN_DEG   = 2.0
SIDE_BUFFER_M    = 3.0
FIXED_HEIGHT_M   = 2.5

# Resolve Street View camera ground elevation from the official Thuringia
# DGM1. Tiles are fetched, unzipped, sampled, and cached only in RAM; no DGM
# files are written to disk. DGM camera heights are used only when a robust
# comparison against the model's individual base vertices is consistent.
ENABLE_DGM_CAMERA_ELEVATION = True
DGM1_TILE_URL_TEMPLATE = (
    "https://geoportal.geoportal-th.de/hoehendaten/DGM/"
    "dgm_2020-2025/"
    "dgm1_32_{easting_km}_{northing_km}_1_th_2020-2025.zip"
)
DGM1_HTTP_TIMEOUT_S = 30.0
DGM1_MAX_MEMORY_TILES = 4
DGM1_EXPECTED_HORIZONTAL_EPSG = 25832
DGM1_EXPECTED_VERTICAL_EPSG = 7837
DGM_BASE_MIN_INLIER_VERTICES = 3
DGM_BASE_MIN_INLIER_FRACTION = 0.66
DGM_BASE_OUTLIER_MAD_SCALE = 3.5
DGM_BASE_OUTLIER_MIN_DEVIATION_M = 0.50
DGM_BASE_MAX_INLIER_ABS_DIFFERENCE_M = 0.75
DGM_BASE_MAX_MEDIAN_ABS_DIFFERENCE_M = 0.50

COVER_MARGIN_PX   = 20
ANGLE_MARGIN_DEG  = 3.0

LR_BAND_BUFFER_PX = 20
LR_BAND_PROTECT_SELECTED_SEGMENTATION = True
LR_BAND_SELECTED_SEGMENT_MARGIN_PX = 20

GRID_OFFSET_M = 20
GRID_N        = 10

# Persistent Street View response/image caching is off by default. Enable it
# only when your Google Maps Platform terms and application policy permit it.
ENABLE_STREETVIEW_CACHE = False
STREETVIEW_CACHE_DIR = r"cache/streetview"
# Restrict discovery to outdoor collections, then require Google-owned
# imagery. This excludes user-contributed panoramas whose position or
# orientation metadata may not be reliable enough for metric projection.
STREETVIEW_SEARCH_SOURCE = "outdoor"
STREETVIEW_GOOGLE_IMAGERY_ONLY = True

PIXELS_PER_METER   = 100.0
MARGIN_METERS      = 0.25
CROP_TO_ALPHA_BBOX = True
FLIP_VERTICAL      = "auto"
OUR_ORDER          = ['t1','t2','b2','b1']

transformer = pyproj.Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
back_tx     = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
EXTRUSION_LEN_XY = 1000.0
BACK_EPS         = 1.0
MAX_ORTHO_PIXELS  = 30_000_000

SAVE_SV_RGB_PER_WALL = False
SAVE_VIEWER_INDEX_JSON = True
SAVE_RAW_OVERLAY_PNG = False
SAVE_LR_OVERLAY_PNG = False
SAVE_WALL_ARTIFACT_FOLDERS = True
WALL_ARTIFACT_FOLDER_NAME = "wall_artifacts"
SAVE_ARTIFACT_CONTACT_SHEET = True
# Fixed orthographic direction for the stage-00 model panel. Keeping this
# independent of each wall's Street View camera makes every contact sheet use
# the same isometric model orientation.
CONTACT_SHEET_MODEL_VIEW_DIRECTION = (0.9, -1.0, -0.75)
SAVE_TEMP_GLOBAL_WALL_GROUP_IMAGE_PROJECTIONS = True
TEMP_GLOBAL_WALL_GROUP_IMAGE_EXPORT_FOLDER = "wall_group_image_projections"
TEMP_GLOBAL_WALL_GROUP_IMAGE_STAGING_FOLDER = "_tmp_wall_group_image_projections"
SAVE_MODEL_DEPTH_MAPS = True
MODEL_DEPTH_NEAR_M = 0.05
MODEL_DEPTH_MAX_MM_PNG = 65535
SOURCE_CRS = "EPSG:25832"
GLB_EXPORT_LOCAL_COORDINATES = True
EXPORT_KMZ = True
KML_ALTITUDE_MODE = "relativeToGround"
SAVE_FACADE_GROUP_DEBUG_PNG = True
FACADE_GROUP_ALLOW_PER_WALL_FALLBACK = False
FACADE_GROUP_MAX_LINE_DEVIATION_M = 1.25
FACADE_GROUP_MAX_LINE_RMS_M = 0.55
FACADE_GROUP_MAX_SEGMENT_ANGLE_DEG = 38.0
FACADE_GROUP_MAX_NORMAL_ANGLE_DEG = 60.0
FACADE_GROUP_MIN_SEGMENT_LENGTH_M = 0.35
# Source selection policy:
# - "auto": use the original per-wall prism winner when every facade group in
#   the building is a single wall; keep projected-coverage selection for real
#   multi-wall facade groups.
# - "projected_coverage": always use the newer grouped-facade selector.
# - "legacy_wall_prism": always prefer the original per-wall prism winner.
FACADE_SOURCE_SELECTION_MODE = "auto"
# Native Street View candidate discovery and projection settings.
FACADE_GROUP_MAX_CANDIDATE_PANOS = 6
FACADE_GROUP_CANDIDATE_TARGET_SPACING_M = 8.0
FACADE_GROUP_CANDIDATE_MIN_FORWARD_M = 2.0
FACADE_GROUP_CANDIDATE_MAX_LATERAL_OUTSIDE_M = 8.0
# Only groups with no genuinely frontal source use this wider metadata
# search. It recovers side facades whose nearest road lies beyond the initial
# building-centered search band without changing established candidate sets.
FACADE_GROUP_RECOVERY_ENABLED = True
FACADE_GROUP_RECOVERY_FORWARD_DISTANCES_M = (20.0, 40.0, 60.0, 80.0)
FACADE_GROUP_RECOVERY_LATERAL_PAD_M = 45.0
FACADE_GROUP_RECOVERY_QUERY_RADIUS_M = 35.0
FACADE_GROUP_RECOVERY_MIN_FRONTALITY = 0.20
FACADE_GROUP_SOURCE_FOV = 100.0
FACADE_PROJECTION_NEAR_PLANE_M = 0.75
FACADE_MAX_PROJECTION_SPAN_FACTOR = 4.0
# A facade that projects to only a line is not a usable texture source even if
# every projected vertex technically lies inside the image.
FACADE_SOURCE_MIN_PROJECTED_AREA_FRACTION = 0.001
FACADE_SOURCE_MIN_PROJECTED_SPAN_PX = 3.0
# Rank source cameras by whether the target wall survives the complete model's
# z-buffer. This detects self-occlusion by nearer walls before image processing.
ENABLE_FACADE_SOURCE_MODEL_VISIBILITY = True
FACADE_SOURCE_VISIBILITY_RENDER_MAX_DIM_PX = 320
FACADE_SOURCE_VISIBILITY_DEPTH_TOLERANCE_M = 0.05
FACADE_SOURCE_VISIBILITY_MASK_ERODE_PX = 1
# Rasterized shared edges can differ by a pixel, so 99.9% is treated as
# geometrically complete visibility after the one-pixel boundary erosion.
FACADE_SOURCE_VISIBILITY_COMPLETE_THRESHOLD = 0.999

# Use nearby OpenStreetMap building footprints to measure how much of each
# corrected target wall is hidden by another building. Every valid candidate is
# globally fitted first, then ranked by its net visible wall fraction after
# frame loss, self-occlusion, and OSM obstruction. If the selected source is
# obstructed, the obstruction-facing divider is extended from image top to
# bottom as an LR-style side crop and the depth-global fit is run again. That
# excluded side is then passed downstream as missing alpha for LaMa to fill.
ENABLE_OSM_EXTERNAL_BUILDING_OCCLUSION = True
OSM_BUILDING_QUERY_RADIUS_M = 120.0
OSM_BUILDING_DEFAULT_HEIGHT_M = 15.0
OSM_BUILDING_LEVEL_HEIGHT_M = 3.0
OSM_BUILDING_DEPTH_TOLERANCE_M = 0.10
OSM_BUILDING_CORRIDOR_BUFFER_M = 1.0
OSM_BUILDING_CLEAR_OCCLUSION_FRACTION = 0.005
OSM_BUILDING_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OSM_BUILDING_OVERPASS_TIMEOUT_S = 90.0
OSM_BUILDING_CACHE_DIR = "cache/osm_building_occlusion"
OSM_BUILDING_REFRESH_CACHE = False
SAVE_OSM_BUILDING_OCCLUSION_DEBUG = True

# Shape-preserving image-space correction of facade-group projections.
ENABLE_FACADE_WIREFRAME_FIT = True
FACADE_WIREFRAME_FIT_RAW_SOURCES = True
FACADE_WIREFRAME_FIT_ALLOW_ROTATION = False
FACADE_WIREFRAME_FIT_MIN_SCORE_IMPROVEMENT = 0.025
FACADE_WIREFRAME_FIT_REFINE_CAMERA_PARAMETERS = True

# Legacy experimental stage that moved the production wireframe toward a SAM
# region. It is disabled because the segmentation has no reliable one-to-one
# relation with the target 3D wall and must not override the first wireframe fit.
ENABLE_DEPTH_AWARE_REGION_FIT = False
DEPTH_AWARE_REGION_FIT_ALLOW_ROTATION = True
DEPTH_AWARE_REGION_FIT_MAX_WORKING_DIM_PX = 360
DEPTH_AWARE_REGION_FIT_MAX_TRANSLATION_PX = 100.0
DEPTH_AWARE_REGION_FIT_SCALE_MIN = 0.80
DEPTH_AWARE_REGION_FIT_SCALE_MAX = 1.20
DEPTH_AWARE_REGION_FIT_MAX_ROTATION_DEG = 5.0
DEPTH_AWARE_REGION_FIT_SEARCH_MARGIN_PX = 120.0
DEPTH_AWARE_REGION_FIT_MIN_SCORE_IMPROVEMENT = 0.035
DEPTH_AWARE_REGION_FIT_MIN_IOU_IMPROVEMENT = 0.025
DEPTH_AWARE_REGION_FIT_MIN_BOUNDARY_IMPROVEMENT = 0.035
DEPTH_AWARE_REGION_FIT_MIN_FINAL_IOU = 0.30
DEPTH_AWARE_REGION_FIT_MIN_FINAL_PRECISION = 0.55
DEPTH_AWARE_REGION_FIT_DEPTH_TOLERANCE_M = 0.08
SAVE_DEPTH_AWARE_REGION_FIT_DEBUG = True

# Choose which accepted image-space correction supplies the wall projection for
# LR cropping, rectification, and all later texture stages. Valid values are
# "depth_global" and "wall_only". A failed/rejected depth fit safely falls back
# to wall-only for that facade group.
FACADE_ALIGNMENT_MODE = "depth_global"

# The selected camera still renders whole-model depth for visibility and
# occlusion. Global alignment uses visible projected model edges with semantic
# priorities; the depth silhouette remains the fallback and diagnostic outline.
# Class contributions are normalized by their total visible length, so a long
# base cannot outweigh a shorter roof merely because it supplies more samples.
ENABLE_MODEL_DEPTH_BOUNDARY_FIT = True
MODEL_DEPTH_BOUNDARY_FIT_ALLOW_ROTATION = False
MODEL_DEPTH_BOUNDARY_FIT_MIN_SCORE_IMPROVEMENT = 0.025
MODEL_DEPTH_BOUNDARY_FIT_MIN_AREA_PX = 350
MODEL_DEPTH_BOUNDARY_FIT_MIN_COMPONENT_FRACTION = 0.02
MODEL_DEPTH_BOUNDARY_FIT_CONTOUR_EPSILON_PX = 1.5
MODEL_DEPTH_BOUNDARY_FIT_MAX_POINTS = 240
# Exact raster-frame closures are excluded before simplification. This smaller
# legacy tolerance is only for old serialized contours that lack provenance.
MODEL_DEPTH_BOUNDARY_IMAGE_BORDER_EPSILON_PX = 0.5
MODEL_DEPTH_BOUNDARY_USE_SEMANTIC_GUIDES = True
MODEL_DEPTH_BOUNDARY_ROOF_WEIGHT = 3.0
MODEL_DEPTH_BOUNDARY_WALL_WEIGHT = 2.0
MODEL_DEPTH_BOUNDARY_BASE_WEIGHT = 0.35
# Image-content guidance for the same depth-global optimizer. SAM3 computes one
# image embedding and evaluates this fixed internal prompt library; users do
# not provide prompts per image. The original rendered model projection is the
# association anchor, so masks from other buildings cannot relocate the fit.
ENABLE_MODEL_DEPTH_PREFIT_SEMANTIC_GUIDANCE = True
MODEL_DEPTH_PREFIT_SEMANTIC_PROMPT_LIBRARY = {
    "building": ("building",),
    "roof": ("roof",),
    "sky": ("sky",),
    "vegetation": ("tree", "bush"),
    "ground": ("road", "ground"),
    "occluder": ("vehicle",),
}
MODEL_DEPTH_PREFIT_SEARCH_MARGIN_PX = 96
MODEL_DEPTH_PREFIT_ASSOCIATION_MARGIN_PX = 48
MODEL_DEPTH_PREFIT_OCCLUDER_DILATION_PX = 7
MODEL_DEPTH_PREFIT_INTERFACE_DILATION_PX = 5
MODEL_DEPTH_PREFIT_BOUNDARY_THICKNESS_PX = 2
# Do not let Canny/LSD or SAM boundary maps score the viewport itself.
MODEL_DEPTH_PREFIT_IMAGE_BORDER_EXCLUSION_PX = 2
MODEL_DEPTH_PREFIT_MIN_INSTANCE_AREA_PX = 80
MODEL_DEPTH_PREFIT_MIN_TARGET_PROJECTION_OVERLAP = 0.01
MODEL_DEPTH_PREFIT_MAX_TARGET_INSTANCES = 4
MODEL_DEPTH_BOUNDARY_SEMANTIC_IMAGE_WEIGHT = 1.35
MODEL_DEPTH_BOUNDARY_SEMANTIC_IMAGE_SIGMA_PX = 6.0
MODEL_DEPTH_BOUNDARY_MAX_SEMANTIC_SCORE_DROP = 0.03
MODEL_DEPTH_BOUNDARY_MIN_MASKED_EVIDENCE_SAMPLES = 8
SAVE_MODEL_DEPTH_PREFIT_SEMANTIC_DEBUG = True
MODEL_DEPTH_BOUNDARY_SEMANTIC_SAMPLE_STEP_PX = 2.0
MODEL_DEPTH_BOUNDARY_SEMANTIC_SILHOUETTE_TOLERANCE_PX = 4.0
MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_SEARCH_RADIUS_PX = 2
MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_TOLERANCE_M = 0.35
MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_RELATIVE_TOLERANCE = 0.03
MODEL_DEPTH_BOUNDARY_SEMANTIC_MAX_GAP_SAMPLES = 2
MODEL_DEPTH_BOUNDARY_SEMANTIC_MIN_RUN_PX = 8.0
SAVE_MODEL_DEPTH_BOUNDARY_FIT_DEBUG = True
MODEL_DEPTH_BOUNDARY_OVERLAY_LINE_THICKNESS_PX = 1
MODEL_DEPTH_BOUNDARY_OVERLAY_DASH_LENGTH_PX = 7
MODEL_DEPTH_BOUNDARY_OVERLAY_DASH_GAP_PX = 5

# After projection crop and rectification, bounded Hough alignment runs first on
# RGB facade edges. SAM then runs on that Hough-adjusted image and is clipped to
# the rectified projection; the optional guarded fit remains the final refinement.
# SAM can never erase the projection-defined facade texture.
ENABLE_POST_RECTIFICATION_SAM = True
POST_RECTIFICATION_SAM_MIN_PIXELS = 250
POST_RECTIFICATION_SAM_MIN_WALL_COVERAGE = 0.15
POST_RECTIFICATION_SAM_MAX_SCALE_DELTA = 0.08
POST_RECTIFICATION_SAM_MAX_TRANSLATION_PX = 30.0

ENABLE_LAMA_FILL = True
LAMA_MODEL_PATH = r"../lama_model/inpainting_lama_2025jan.onnx"
LAMA_MASK_DILATE_PX = 5
LAMA_MIN_HOLE_AREA_PX = 64
LAMA_SAVE_DEBUG_MASK = True

ENABLE_MULTI_FACADE_INSTANCE_SELECTION = True
FACADE_INSTANCE_MAX_SELECTED = 4
FACADE_INSTANCE_MIN_INSIDE_RATIO = 0.45
FACADE_INSTANCE_MAX_OUTSIDE_RATIO = 0.55
FACADE_INSTANCE_MIN_NEW_WALL_COVER = 0.04
FACADE_INSTANCE_MIN_NEW_INSIDE_PX = 500
FACADE_INSTANCE_MAX_DUPLICATE_INSIDE_RATIO = 0.75
FACADE_INSTANCE_MAX_CENTER_DIST = 0.65

# Optional legacy refinement: uniformly scale/translate the bounded SAM contour
# inside the already rectified wall polygon. Depth-global alignment is now the
# authoritative geometry, so keep this off unless explicitly experimenting.
ENABLE_ORTHO_FIT = False
ENABLE_ORTHO_POLYGON_FIT = ENABLE_ORTHO_FIT  # compatibility alias for older scripts
POLYGON_FIT_MAX_CONTOUR_POINTS = 450
POLYGON_FIT_CENTER_SHIFT_STEPS = 13
POLYGON_FIT_BINARY_STEPS = 28
PERSPECTIVE_FIT_CENTER_SHIFT_FRAC = 0.20
PERSPECTIVE_FIT_CENTER_SHIFT_STEPS = 21
PERSPECTIVE_FIT_BINARY_STEPS = 40
PERSPECTIVE_FIT_INSET_PX = 0.25
PERSPECTIVE_FIT_SCALE_GROWTH = 1.35
PERSPECTIVE_FIT_MAX_SCALE = 6.0
QUAD_MIN_COMPONENT_AREA_PX = 200
QUAD_MORPH_CLOSE_PX = 7
QUAD_MORPH_OPEN_PX = 3
QUAD_FILL_HOLES = True
QUAD_MIN_CONTOUR_AREA_PX = 500
FIT_CLIP_TO_WALL = True

# This is both the bounded Hough diagnostic and its guided warp stage. In the
# grouped production path it runs before post-rectification SAM.
ENABLE_ORTHO_HOUGH_DEBUG = True
HOUGH_SEARCH_BAND_PX = 80
HOUGH_MIN_LENGTH_PX = 120
HOUGH_MAX_GAP_PX = 20
HOUGH_ANGLE_THRESH_DEG = 12.0
HOUGH_CANNY_LOW = 50
HOUGH_CANNY_HIGH = 150
HOUGH_CANNY_DILATE_PX = 1
HOUGH_USE_CLAHE = True
HOUGH_SAVE_BAND_MASKS = False

ENABLE_HOUGH_GUIDED_WARP = True
# When just one facade side is detected, hold the missing side fixed and
# progressively warp toward the detected side instead of discarding the cue.
ENABLE_HOUGH_SINGLE_SIDE_WARP = True
SAVE_HOUGH_WARP_DEBUG = True

NAMING_STYLE = "legacy"

STAGE_PATTERNS = {
    "legacy": {
        "raw_overlay":      "{base}_wall{wall:02d}_overlay.png",
        "lr_band_overlay":  "{base}_wall{wall:02d}_lr_overlay.png",
        "sam3_overlay":     "{base}_wall{wall:02d}_sam3_overlay.png",
        "ortho_png":        "{base}_wall{wall:02d}_ortho.png",
        "ortho_overlay":    "{base}_wall{wall:02d}_ortho_overlay.png",
        "ortho_meta":       "{base}_wall{wall:02d}_ortho.json",
        "glb":              "{base}__textured.glb",
        "kmz":              "{base}__textured.kmz",
        "sam3_instances_overlay": "{base}_wall{wall:02d}_sam3_instances_overlay.png",
        "ortho_prefit_overlay": "{base}_wall{wall:02d}_ortho_prefit_overlay.png",
        "ortho_fit_overlay": "{base}_wall{wall:02d}_ortho_fit_overlay.png",
        "hough_overlay": "{base}_wall{wall:02d}_hough_overlay.png",
        "hough_warp_overlay": "{base}_wall{wall:02d}_hough_warp_overlay.png",
    },
    "verbose": {
        "raw_overlay":      "{wallbase}__overlay.png",
        "lr_band_overlay":  "{wallbase}__lr_band_rgba__overlay.png",
        "sam3_overlay":     "{wallbase}__sam3_building_rgba__overlay.png",
        "ortho_png":        "{wallbase}__ortho_final_rgba.png",
        "ortho_overlay":    "{wallbase}__ortho_final_rgba__overlay.png",
        "ortho_meta":       "{wallbase}__ortho_final_meta.json",
        "glb":              "{base}__textured.glb",
        "kmz":              "{base}__textured.kmz",
        "sam3_instances_overlay": "{wallbase}__sam3_instances_overlay.png",
        "ortho_prefit_overlay": "{wallbase}__ortho_prefit_rgba__overlay.png",
        "ortho_fit_overlay": "{wallbase}__ortho_fit_overlay.png",
        "hough_overlay": "{wallbase}__hough_overlay.png",
        "hough_warp_overlay": "{wallbase}__hough_warp_overlay.png",
    },
}


def _apply_local_overrides():
    try:
        from lod2_texture_pipeline.config_local import LOCAL_CONFIG
    except ImportError:
        return

    for key, value in LOCAL_CONFIG.items():
        if key.isupper():
            globals()[key] = value

    if "ENABLE_ORTHO_FIT" not in LOCAL_CONFIG and "ENABLE_ORTHO_POLYGON_FIT" in LOCAL_CONFIG:
        globals()["ENABLE_ORTHO_FIT"] = bool(LOCAL_CONFIG["ENABLE_ORTHO_POLYGON_FIT"])
    globals()["ENABLE_ORTHO_POLYGON_FIT"] = globals().get("ENABLE_ORTHO_FIT", False)


_apply_local_overrides()
