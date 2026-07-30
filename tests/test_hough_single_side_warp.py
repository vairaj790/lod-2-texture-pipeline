import numpy as np
import pytest

import lod2_texture_pipeline.pipeline as pipeline
from lod2_texture_pipeline.quadfit import apply_hough_guided_ortho_warp


def _vertical_line(x, height):
    return np.array([[float(x), 0.0], [float(x), float(height - 1)]])


@pytest.mark.parametrize(
    ("detected_side", "source_x", "target_x", "fixed_x"),
    [
        ("right", 80, 90, 10),
        ("left", 20, 10, 90),
    ],
)
def test_hough_warp_uses_missing_side_as_identity_anchor(
    detected_side,
    source_x,
    target_x,
    fixed_x,
):
    height = 80
    width = 100
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, source_x - 1:source_x + 2, 0] = 255
    rgba[:, source_x - 1:source_x + 2, 3] = 255
    rgba[:, fixed_x - 1:fixed_x + 2, 1] = 255
    rgba[:, fixed_x - 1:fixed_x + 2, 3] = 255

    proj_left = _vertical_line(10, height)
    proj_right = _vertical_line(90, height)
    selected_left = (
        _vertical_line(source_x, height) if detected_side == "left" else None
    )
    selected_right = (
        _vertical_line(source_x, height) if detected_side == "right" else None
    )

    warped = apply_hough_guided_ortho_warp(
        ortho_rgba=rgba,
        sel_left_line=selected_left,
        sel_right_line=selected_right,
        sel_top_line=None,
        proj_left_line=proj_left,
        proj_right_line=proj_right,
        proj_top_line=None,
    )

    red_profile = warped[:, :, 0].sum(axis=0)
    green_profile = warped[:, :, 1].sum(axis=0)
    assert abs(int(np.argmax(red_profile)) - target_x) <= 1
    assert abs(int(np.argmax(green_profile)) - fixed_x) <= 1


def test_hough_warp_without_any_side_line_is_identity():
    rgba = np.arange(20 * 30 * 4, dtype=np.uint8).reshape(20, 30, 4)
    warped = apply_hough_guided_ortho_warp(
        ortho_rgba=rgba,
        sel_left_line=None,
        sel_right_line=None,
        sel_top_line=None,
        proj_left_line=_vertical_line(3, 20),
        proj_right_line=_vertical_line(26, 20),
        proj_top_line=None,
    )

    np.testing.assert_array_equal(warped, rgba)


def test_disabled_ortho_fit_keeps_sam_contour_for_debug(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_ORTHO_FIT", False)
    rgba = np.zeros((80, 100, 4), dtype=np.uint8)
    rgba[3:77, 3:97, 3] = 255
    sam_mask = np.zeros((80, 100), dtype=bool)
    sam_mask[10:70, 20:80] = True
    wall_poly = np.array([[2, 77], [97, 77], [97, 2], [2, 2]], dtype=np.float64)

    result, transform, source_pts, fitted_pts, info = (
        pipeline._fit_ortho_rgba_alpha_inside_polygon(
            rgba,
            wall_poly,
            source_mask_override=sam_mask,
        )
    )

    np.testing.assert_array_equal(result, rgba)
    assert transform is None
    assert source_pts is not None
    assert fitted_pts is None
    assert info["applied"] is False
    assert info["reason"] == "disabled"
    assert info["source_mask"] == "post_rectification_sam_inside_projection"
    assert info["source_area_px"] == int(sam_mask.sum())
