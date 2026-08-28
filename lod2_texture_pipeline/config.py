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
# Final-export geometry repair. Texture generation continues to use the
# original LiDAR wall geometry; only the finished export receives wall skirts
# down to the building-wide minimum base elevation and a flat base. Textured
# skirts reuse the source wall appearance directly above their join.
ENABLE_POSTTEXTURE_BASE_LEVEL_REPAIR = True
POSTTEXTURE_BASE_LEVEL_TOLERANCE_M = 0.001
POSTTEXTURE_EXTENSION_DOMINANT_COLOR_BITS = 5
POSTTEXTURE_EXTENSION_MAX_COLOR_SAMPLES = 500_000
POSTTEXTURE_EXTENSION_SEAM_BAND_FRACTION = 0.04
POSTTEXTURE_EXTENSION_SEAM_BAND_MIN_TEXELS = 8.0
POSTTEXTURE_EXTENSION_SEAM_BAND_MAX_TEXELS = 24.0
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
# corrected target wall is hidden by another building. Raw-canvas OSM blockers
# are excluded before SAM association and fitting; nearly fully blocked views
# are rejected before SAM3. Remaining candidates are ranked by net visible wall
# fraction after frame loss, self-occlusion, and OSM obstruction. If selected is
# obstructed, the obstruction-facing divider is extended from image top to
# bottom as an LR-style side crop and the depth-global fit is run again. That
# excluded side is then passed downstream as missing alpha for LaMa to fill.
ENABLE_OSM_EXTERNAL_BUILDING_OCCLUSION = True
OSM_BUILDING_QUERY_RADIUS_M = 120.0
OSM_BUILDING_DEFAULT_HEIGHT_M = 15.0
OSM_BUILDING_LEVEL_HEIGHT_M = 3.0
# Neighbouring footprints must be extruded from their own terrain elevation.
# A shared target-building base is invalid on steep streets and can bury an
# otherwise tall foreground blocker below the camera rays.
OSM_BUILDING_USE_DGM_TERRAIN = True
OSM_BUILDING_TERRAIN_MAX_SAMPLES = 9
OSM_BUILDING_TERRAIN_TOP_MARGIN_M = 0.5
OSM_BUILDING_DEPTH_TOLERANCE_M = 0.10
OSM_BUILDING_CORRIDOR_BUFFER_M = 1.0
# Rendering/depth overlap is the authoritative image-space test.  Requiring a
# narrow ground-plan corridor first can discard real blockers when Street View
# positions, roof overhangs, or footprints differ by a few metres.
OSM_BUILDING_REQUIRE_CORRIDOR_INTERSECTION = False
OSM_BUILDING_CLEAR_OCCLUSION_FRACTION = 0.005
# A raw or fitted target with at least this much OSM blockage cannot be a source.
# The high threshold preserves partially visible views whose small anchored fit
# can improve alignment, while eliminating the observed 98-100% blocked cases.
OSM_BUILDING_PREFIT_HARD_REJECT_FRACTION = 0.97
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
# Keep the global model around its original projection. Strong target semantics
# allow the checked +/-50 px corrections; missing/weak semantics only permit a
# micro correction. All pixel limits are specified for a 640 px canvas and are
# scaled with image size inside the fitter.
MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_SCALE_DELTA = 0.10
# A wider scale range is available only to the guarded background-aware
# challenger. The incumbent keeps the stable 0.90-1.10 envelope, and the wider
# result is selected only after a material improvement against the same full
# roof/sky guide.
MODEL_DEPTH_BACKGROUND_AWARE_MAX_SCALE_DELTA = 0.25
MODEL_DEPTH_BACKGROUND_AWARE_MAX_LEGACY_ROOF_RETENTION = 0.50
MODEL_DEPTH_BACKGROUND_AWARE_MIN_RESTORED_ROOF_PIXELS = 64
MODEL_DEPTH_BACKGROUND_AWARE_MIN_FULL_ROOF_SCORE_GAIN = 0.08
MODEL_DEPTH_BACKGROUND_AWARE_MAX_COMMON_SEMANTIC_SCORE_DROP = 0.02
MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_TRANSLATION_PX = 50.0
MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_MEAN_DISPLACEMENT_PX = 55.0
MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_TRANSLATION_FRACTION = 0.25
MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_DISPLACEMENT_FRACTION = 0.25
MODEL_DEPTH_BOUNDARY_ANCHOR_MIN_IOU = 0.35
MODEL_DEPTH_BOUNDARY_ANCHOR_TRANSLATION_PRIOR_WEIGHT = 0.40
MODEL_DEPTH_BOUNDARY_ANCHOR_TRANSLATION_PRIOR_SIGMA_PX = 40.0
MODEL_DEPTH_BOUNDARY_MICRO_MAX_SCALE_DELTA = 0.06
MODEL_DEPTH_BOUNDARY_MICRO_MAX_TRANSLATION_PX = 20.0
MODEL_DEPTH_BOUNDARY_MICRO_MAX_MEAN_DISPLACEMENT_PX = 20.0
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
    # Fit-only roof semantics.  The broader bare-roof prompt remains separate
    # below so vehicle roofs cannot become fitting guides while post-Hough
    # cleanup retains its established evidence.
    "roof": ("building roof",),
    "sky": ("sky",),
    "vegetation": ("tree", "bush", "hedge"),
    "ground": ("road", "ground"),
    "occluder": (
        "vehicle",
        "person",
        "traffic sign",
        "signboard",
        "advertising board",
        "billboard",
        "street furniture",
        "pole",
        "scaffolding",
        "retaining wall",
        "garden boundary wall",
        "fence",
        "gate",
    ),
    # Broad, category-independent proposals are kept separate from the hard
    # prompt library and must pass component-size/target-overlap safeguards.
    "generic_occluder": (
        "foreground object",
        "object in front of building",
    ),
}
MODEL_DEPTH_PREFIT_DOWNSTREAM_ROOF_PROMPTS = ("roof",)
# Require a fit roof to agree with the projected top and selected building, and
# consume only a verified roof-to-sky/background-vegetation interface outside
# foreground guards.  Inferred occlusion bridges are diagnostic-only.
ENABLE_MODEL_DEPTH_PREFIT_STRICT_ROOF_GUIDANCE = True
MODEL_DEPTH_PREFIT_STRICT_ROOF_BAND_RADIUS_PX = 18
MODEL_DEPTH_PREFIT_STRICT_ROOF_UPPER_BUILDING_FRACTION = 0.48
MODEL_DEPTH_PREFIT_STRICT_ROOF_ATTACHMENT_RADIUS_PX = 8
MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_BAND_PIXELS = 12
MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_BAND_SPAN_FRACTION = 0.03
MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_ATTACHMENT_PIXELS = 12
MODEL_DEPTH_PREFIT_STRICT_ROOF_MAX_FOREGROUND_FRACTION = 0.35
MODEL_DEPTH_PREFIT_STRICT_ROOF_CONTEXT_RADIUS_PX = 3
MODEL_DEPTH_PREFIT_STRICT_ROOF_FOREGROUND_GUARD_RADIUS_PX = 4
MODEL_DEPTH_PREFIT_STRICT_ROOF_VEGETATION_INSET_PX = 2
MODEL_DEPTH_PREFIT_STRICT_ROOF_VEGETATION_INSIDE_OFFSET_PX = 8
MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_GUIDE_COMPONENT_PIXELS = 5
ENABLE_MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_DIAGNOSTIC = True
MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_MIN_ENDPOINT_RUN_PX = 3
MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_MAX_GAP_PX = 64
MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_DOMAIN_DILATION_PX = 2
MODEL_DEPTH_PREFIT_SEARCH_MARGIN_PX = 96
MODEL_DEPTH_PREFIT_ASSOCIATION_MARGIN_PX = 48
# Candidate visibility is measured on the target wall with a tighter search;
# this avoids counting trees elsewhere in the whole-model 96 px search region.
MODEL_DEPTH_PREFIT_TARGET_WALL_SEARCH_MARGIN_PX = 32
MODEL_DEPTH_PREFIT_TARGET_WALL_ASSOCIATION_MARGIN_PX = 24
MODEL_DEPTH_PREFIT_OCCLUDER_DILATION_PX = 3
MODEL_DEPTH_PREFIT_INTERFACE_DILATION_PX = 5
MODEL_DEPTH_PREFIT_BOUNDARY_THICKNESS_PX = 2
# Do not let Canny/LSD or SAM boundary maps score the viewport itself.
MODEL_DEPTH_PREFIT_IMAGE_BORDER_EXCLUSION_PX = 2
MODEL_DEPTH_PREFIT_MIN_INSTANCE_AREA_PX = 80
MODEL_DEPTH_PREFIT_MIN_TARGET_PROJECTION_OVERLAP = 0.01
MODEL_DEPTH_PREFIT_MAX_TARGET_INSTANCES = 4
# High-confidence candidate rejection using the same SAM3 embedding/masks (no
# second model inference). Missing SAM falls back to geometry+OSM; successful
# segmentation must show meaningful building/roof support on the target wall.
MODEL_DEPTH_PREFIT_VISIBILITY_MIN_TARGET_PIXELS = 250
MODEL_DEPTH_PREFIT_VISIBILITY_MIN_TARGET_SUPPORT_FRACTION = 0.10
MODEL_DEPTH_PREFIT_VISIBILITY_MAX_OCCLUDER_FRACTION = 0.80
MODEL_DEPTH_PREFIT_VISIBILITY_LOW_SUPPORT_OCCLUDER_FRACTION = 0.60
MODEL_DEPTH_PREFIT_VISIBILITY_MIN_LARGEST_COMPONENT_FRACTION = 0.05
# With an anchored fit, a SAM building region many times larger than the raw
# whole-model projection is almost certainly a foreground building that merely
# overlaps the anchor, not the distant target.
MODEL_DEPTH_PREFIT_VISIBILITY_MAX_TARGET_AREA_RATIO = 6.0
MODEL_DEPTH_PREFIT_VISIBILITY_REJECT_NO_TARGET = True
# Category-independent safeguard for foreground objects that are not returned
# by one of the explicit prompts. Only small/medium non-target components well
# inside a sufficiently supported raw projection suppress fitting evidence;
# large residuals fall back to the normal target/geometry fit.
ENABLE_MODEL_DEPTH_PREFIT_GENERIC_NON_TARGET = True
MODEL_DEPTH_PREFIT_GENERIC_MIN_TARGET_COVERAGE = 0.20
MODEL_DEPTH_PREFIT_GENERIC_PROJECTION_INSET_PX = 3
MODEL_DEPTH_PREFIT_GENERIC_TARGET_DILATION_PX = 2
MODEL_DEPTH_PREFIT_GENERIC_MIN_COMPONENT_AREA_PX = 80
MODEL_DEPTH_PREFIT_GENERIC_MAX_COMPONENT_FRACTION = 0.20
MODEL_DEPTH_PREFIT_GENERIC_MAX_TOTAL_FRACTION = 0.45
MODEL_DEPTH_PREFIT_GENERIC_MAX_TARGET_OVERLAP_FRACTION = 0.15
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

# Reuse the selected candidate's full-image SAM3 building/occluder evidence for
# facade extraction. The mask is carried through rectification, the exact Hough
# remap, and the optional guarded affine adjustment; no second SAM3 inference is
# run on the rectified wall.
ENABLE_PREFIT_SEMANTIC_TEXTURE_MASK_REUSE = True
PREFIT_SEMANTIC_TEXTURE_MIN_PIXELS = 250
PREFIT_SEMANTIC_TEXTURE_MIN_WALL_COVERAGE = 0.35
PREFIT_SEMANTIC_TEXTURE_CLOSE_PX = 2
PREFIT_SEMANTIC_TEXTURE_MAX_HOLE_AREA_PX = 900
# If a hard-prompt union covers nearly the whole fitted wall, treat it as a
# likely broad-prompt failure in the projection fallback instead of returning
# an empty facade. OSM exclusions remain authoritative and are never relaxed.
PREFIT_SEMANTIC_TEXTURE_MAX_HARD_EXCLUSION_FRACTION = 0.85
PREFIT_SEMANTIC_TEXTURE_MAX_SCALE_DELTA = 0.08
PREFIT_SEMANTIC_TEXTURE_MAX_TRANSLATION_PX = 30.0
SAVE_PREFIT_SEMANTIC_TEXTURE_REUSE_DEBUG = True

# Remove roof-class pixels that remain inside a rectified facade after the
# Hough warp. A roof that topologically separates the full wall polygon into
# upper and lower partitions also removes the lower, foreground structure;
# an isolated roof/awning removes only its own pixels. The split test uses the
# complete wall polygon, so pre-existing texture holes cannot trigger it.
ENABLE_POST_HOUGH_ROOF_STRUCTURE_REMOVAL = True
POST_HOUGH_ROOF_CONNECTION_TOLERANCE_PX = 3
POST_HOUGH_ROOF_BOUNDARY_SEED_PX = 2
POST_HOUGH_ROOF_MIN_DIVIDER_COMPONENT_AREA_PX = 32
POST_HOUGH_ROOF_MIN_PARTITION_AREA_PX = 80
POST_HOUGH_ROOF_MIN_PARTITION_FRACTION = 0.03

ENABLE_LAMA_FILL = True
LAMA_MODEL_PATH = r"../lama_model/inpainting_lama_2025jan.onnx"
# ``auto`` prefers the official native Big-LaMa generator when its clean state
# file and CUDA are available, and otherwise retains the fixed-512 ONNX path.
LAMA_BACKEND = "auto"  # auto | big_lama | onnx
BIG_LAMA_GENERATOR_PATH = r"../lama_model/big-lama/models/generator.pt"
BIG_LAMA_DEVICE = "auto"  # auto | cuda | cpu
BIG_LAMA_CONTEXT_PX = 128
BIG_LAMA_MAX_SIDE_PX = 1280
BIG_LAMA_MAX_PIXELS = 1_000_000
# Official coarse-to-fine feature refinement is available for the largest
# holes. Keep it opt-in until its extra runtime is acceptable for a deployment.
BIG_LAMA_ENABLE_REFINEMENT = False
BIG_LAMA_REFINEMENT_ITERATIONS = 15
BIG_LAMA_REFINEMENT_LEARNING_RATE = 0.002
BIG_LAMA_REFINEMENT_MIN_SIDE = 512
BIG_LAMA_REFINEMENT_MAX_SCALES = 2
BIG_LAMA_REFINEMENT_MAX_PIXELS = 400_000
# The OpenCV ONNX export has a fixed 512x512 input.  Keep its global pass
# aspect-ratio preserving, then refine masked regions in overlapping native-
# resolution tiles.  This avoids stretching multi-kilopixel facade textures
# through a single square inference.
LAMA_ENABLE_HIGH_RES_TILING = True
LAMA_TILE_OVERLAP_PX = 96
LAMA_MAX_TILES = 64
# Native tiles are used as a detail layer over the globally coherent coarse
# prediction. Removing each tile's low-frequency residual prevents adjacent
# 512px crops from producing visible rectangular color/exposure changes.
LAMA_TILE_DETAIL_SIGMA_PX = 16.0
LAMA_TILE_LOW_FREQUENCY_WEIGHT = 0.0
LAMA_TILE_DETAIL_STRENGTH = 0.75
LAMA_TILE_DETAIL_MAX_DELTA = 40.0
# Zero lets ONNX Runtime choose an appropriate CPU thread pool.  Set to 1 on
# hosts where the runtime's automatic pthread affinity messages are unwanted.
LAMA_ONNX_INTRA_OP_THREADS = 0

# The inference mask includes a small safety ring around invalid pixels.  Only
# the original holes are replaced fully; this ring is feathered into the known
# texture so valid facade detail remains unchanged away from the seam.
LAMA_MASK_DILATE_PX = 8
LAMA_COMPOSITE_FEATHER_PX = 4
LAMA_MIN_HOLE_AREA_PX = 64
LAMA_SAVE_DEBUG_MASK = True

# Keep valid RGB below transparent texels around the wall polygon.  Alpha stays
# zero there, but the gutter prevents bilinear/mipmap sampling from pulling
# black into the visible texture edge in GLB/KMZ viewers.
TEXTURE_TRANSPARENT_EDGE_BLEED_PX = 16

ENABLE_MULTI_FACADE_INSTANCE_SELECTION = True
FACADE_INSTANCE_MAX_SELECTED = 4
FACADE_INSTANCE_MIN_INSIDE_RATIO = 0.45
FACADE_INSTANCE_MAX_OUTSIDE_RATIO = 0.55
FACADE_INSTANCE_MIN_NEW_WALL_COVER = 0.04
FACADE_INSTANCE_MIN_NEW_INSIDE_PX = 500
FACADE_INSTANCE_MAX_DUPLICATE_INSIDE_RATIO = 0.75
FACADE_INSTANCE_MAX_CENTER_DIST = 0.65

# Optional legacy refinement: uniformly scale/translate the retained content
# contour inside the already rectified wall polygon. Depth-global alignment is
# now authoritative, so keep this off unless explicitly experimenting.
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
# grouped production path it also carries the reused full-image semantic mask.
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

# Grouped-facade side lines must describe most of the finite projected wall
# edge, not merely touch its search band.  These are hard acceptance gates;
# the opening-aware solver receives only lines that pass every one.
HOUGH_SIDE_ANGLE_THRESH_DEG = 8.0
HOUGH_SIDE_MIN_TARGET_COVERAGE_RATIO = 0.75
HOUGH_SIDE_MAX_LENGTH_RATIO = 1.20
HOUGH_SIDE_MAX_DISTANCE_PX = 36.0
HOUGH_SIDE_MAX_DISTANCE_TARGET_RATIO = 0.04
HOUGH_SIDE_MIN_BAND_OCCUPANCY_RATIO = 0.80
HOUGH_SIDE_MIN_EDGE_SUPPORT_RATIO = 0.30
HOUGH_SIDE_OFFSET_CLUSTER_PX = 12.0

# Inspect the two facade sides independently before the reusable SAM3 target
# mask is clipped to the model projection.  A foreground tree/object covering
# most of one side disables recovery on that side; sky/background does not.
ENABLE_FACADE_SIDE_SEMANTIC_RECOVERY = True
FACADE_SIDE_SOURCE_BAND_PX = 48
FACADE_SIDE_FOREGROUND_OCCLUSION_RATIO = 0.50
FACADE_SIDE_MIN_ADJACENT_VISIBLE_FRACTION = 0.08
FACADE_SIDE_MIN_SEMANTIC_INTERFACE_SUPPORT = 0.20

# Windows and doors supply the two shared Manhattan directions.  They never
# constrain roof or ground edges, and a wall-side line is admitted only when
# it agrees with the repeated opening orientation.
ENABLE_OPENING_AWARE_RECTIFICATION = True
OPENING_AWARE_PROMPT_LIBRARY = {
    "window": ("window", "building window", "shop window"),
    "door": ("door", "building entrance door"),
}
OPENING_AWARE_PROPOSAL_THRESHOLD = 0.20
OPENING_AWARE_MIN_SAM_SCORE = 0.25
OPENING_AWARE_MIN_STABILITY = 0.78
OPENING_AWARE_MIN_OPENINGS = 3
OPENING_AWARE_MAX_CONSENSUS_DEG = 5.0
OPENING_AWARE_ALLOW_PROJECTIVE = True
OPENING_AWARE_MAX_FINAL_SIDE_ANGLE_DEG = 2.0
OPENING_AWARE_MAX_FINAL_SIDE_DISTANCE_PX = 8.0
OPENING_AWARE_MAX_FINAL_P90_AXIS_ERROR_DEG = 3.0
OPENING_AWARE_MAX_FINAL_P90_ORTHOGONALITY_ERROR_DEG = 5.0
OPENING_AWARE_MAX_FINAL_PER_OPENING_AXIS_ERROR_DEG = 4.0
OPENING_AWARE_MAX_FINAL_PER_OPENING_ORTHOGONALITY_ERROR_DEG = 5.0
SAVE_OPENING_AWARE_DEBUG = True

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
