import cv2
import numpy as np
from PIL import Image

from lod2_texture_pipeline.pipeline import (
    _extract_contour_points_from_mask,
    _save_ortho_fit_debug_overlay,
)


def test_ortho_fit_overlay_does_not_bridge_disconnected_source_contours(tmp_path):
    source_mask = np.zeros((120, 120), dtype=bool)
    source_mask[35:61, 12:43] = True
    source_mask[80:108, 78:111] = True

    rgba = np.zeros((120, 120, 4), dtype=np.uint8)
    rgba[:, :, :3] = 220
    rgba[:, :, 3] = source_mask.astype(np.uint8) * 255
    source_pts, _contours = _extract_contour_points_from_mask(source_mask)
    output_path = tmp_path / "ortho_fit_overlay.png"

    _save_ortho_fit_debug_overlay(
        img_rgba=rgba,
        wall_poly_px=np.array(
            [[5.0, 110.0], [114.0, 110.0], [114.0, 5.0], [5.0, 5.0]],
            dtype=np.float64,
        ),
        source_pts=source_pts,
        fitted_pts=None,
        out_path=str(output_path),
        fit_info={"applied": False, "reason": "disabled"},
        source_mask=source_mask,
    )

    overlay = np.asarray(Image.open(output_path).convert("RGB"))
    orange = (
        (overlay[:, :, 0] > 245)
        & (overlay[:, :, 1] >= 165)
        & (overlay[:, :, 1] <= 195)
        & (overlay[:, :, 2] < 20)
    ).astype(np.uint8)
    component_count, _labels = cv2.connectedComponents(orange, connectivity=8)

    assert component_count - 1 == 2
    assert int(orange[61:82, 48:73].sum()) == 0


def test_ortho_fit_overlay_only_displays_selected_sam_pixels(tmp_path):
    selected_mask = np.zeros((100, 120), dtype=bool)
    selected_mask[25:76, 35:86] = True

    rgba = np.zeros((100, 120, 4), dtype=np.uint8)
    rgba[:, :, :3] = (25, 100, 180)
    rgba[:, :, 3] = 255
    output_path = tmp_path / "combined_sam_fit_overlay.png"

    _save_ortho_fit_debug_overlay(
        img_rgba=rgba,
        wall_poly_px=np.array(
            [[10.0, 90.0], [109.0, 90.0], [109.0, 10.0], [10.0, 10.0]],
            dtype=np.float64,
        ),
        source_pts=None,
        fitted_pts=None,
        out_path=str(output_path),
        fit_info={"applied": False, "reason": "disabled"},
        source_mask=selected_mask,
        display_mask=selected_mask,
    )

    overlay = np.asarray(Image.open(output_path).convert("RGB"))
    np.testing.assert_array_equal(overlay[50, 60], np.array([25, 100, 180]))
    np.testing.assert_array_equal(overlay[50, 20], np.array([246, 246, 244]))
    assert not np.array_equal(overlay[50, 20], np.array([255, 0, 0]))
