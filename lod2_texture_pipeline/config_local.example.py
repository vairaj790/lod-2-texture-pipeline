LOCAL_CONFIG = {
    "GEOJSON_DIR": "/absolute/path/to/3d_geojsons",
    "GEOTIFF_DIR": "/absolute/path/to/geotiffs",
    "OUTPUT_DIR": "/absolute/path/to/outputs",
    "API_KEY": "YOUR_GOOGLE_STREET_VIEW_API_KEY",
    "STREETVIEW_SEARCH_SOURCE": "outdoor",
    "STREETVIEW_GOOGLE_IMAGERY_ONLY": True,
    # Keep persistent API response/image caching disabled unless your Google
    # Maps Platform terms and application policy explicitly permit it.
    "ENABLE_STREETVIEW_CACHE": False,
    "ENABLE_DGM_CAMERA_ELEVATION": True,
    "DGM_BASE_MAX_INLIER_ABS_DIFFERENCE_M": 0.75,
    "DGM_BASE_MAX_MEDIAN_ABS_DIFFERENCE_M": 0.50,
    "ENABLE_FACADE_SOURCE_MODEL_VISIBILITY": True,
    "ENABLE_OSM_EXTERNAL_BUILDING_OCCLUSION": True,
    "OSM_BUILDING_QUERY_RADIUS_M": 120.0,
    "OSM_BUILDING_CLEAR_OCCLUSION_FRACTION": 0.005,
    "ENABLE_ORTHO_FIT": False,
    "ENABLE_HOUGH_SINGLE_SIDE_WARP": True,
    "LAMA_MODEL_PATH": "../lama_model/inpainting_lama_2025jan.onnx",
    "SAM3_PROMPT_FACADE": "building facade",
    "SAM3_PROMPT_FACADE_REFINEMENT": "building walls",
    "SAM3_PROMPT_ROOF": "roof",
    "ENABLE_MODEL_DEPTH_PREFIT_SEMANTIC_GUIDANCE": True,
    "MODEL_DEPTH_PREFIT_IMAGE_BORDER_EXCLUSION_PX": 2,
    "MODEL_DEPTH_BOUNDARY_IMAGE_BORDER_EPSILON_PX": 0.5,
    # Optional fixed, automatic prompt library used before depth-global fits.
    # Add local occluder concepts here without changing pipeline code.
    "MODEL_DEPTH_PREFIT_SEMANTIC_PROMPT_LIBRARY": {
        "building": ("building",),
        "roof": ("roof",),
        "sky": ("sky",),
        "vegetation": ("tree", "bush"),
        "ground": ("road", "ground"),
        "occluder": ("vehicle",),
    },
    "FACADE_ALIGNMENT_MODE": "depth_global",  # or "wall_only"
}
