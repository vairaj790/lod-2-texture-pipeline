# -*- coding: utf-8 -*-
"""Central configuration for the LoD-2 texture pipeline."""

import pyproj

# ======================= USER CONFIG (BATCH) =======================
GEOJSON_DIR = r"sample_data/3d_geojsons/singles"
GEOTIFF_DIR = r"sample_data/geotiffs/singles"
OUTPUT_DIR  = r"outputs"

API_KEY      = ""
MODEL_NAME   = "SAM3_PROMPT"
SAM3_PROMPT_FACADE = "building facade"
SAM3_PROMPT_ROOF   = "roof"
ROOF_SUBTRACT_DILATE_PX = 2

SV_SIZE          = "640x640"
FOV_MIN, FOV_MAX = 15.0, 120.0
FOV_MARGIN_DEG   = 2.0
SIDE_BUFFER_M    = 3.0
FIXED_HEIGHT_M   = 2.5

COVER_MARGIN_PX   = 20
ANGLE_MARGIN_DEG  = 3.0
TILE_FOV          = 90.0
TILE_OVERLAP_DEG  = 20.0

LR_BAND_BUFFER_PX = 20

GRID_OFFSET_M = 20
GRID_N        = 10

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

ENABLE_LAMA_FILL = True
LAMA_MODEL_PATH = r"lama_model/inpainting_lama_2025jan.onnx"
LAMA_MASK_DILATE_PX = 5
LAMA_MIN_HOLE_AREA_PX = 64
LAMA_SAVE_DEBUG_MASK = True

ENABLE_ORTHO_QUAD_FIT = True
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
        "sam3_instances_overlay": "{base}_wall{wall:02d}_sam3_instances_overlay.png",
        "ortho_prefit_overlay": "{base}_wall{wall:02d}_ortho_prefit_overlay.png",
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
        "sam3_instances_overlay": "{wallbase}__sam3_instances_overlay.png",
        "ortho_prefit_overlay": "{wallbase}__ortho_prefit_rgba__overlay.png",
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


_apply_local_overrides()