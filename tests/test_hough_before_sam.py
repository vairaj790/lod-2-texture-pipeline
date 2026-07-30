import numpy as np
from PIL import Image

import lod2_texture_pipeline.pipeline as pipeline


def test_group_hough_without_mask_override_uses_rectified_rgb_edges(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(pipeline, "ENABLE_ORTHO_HOUGH_DEBUG", True)
    monkeypatch.setattr(pipeline, "ENABLE_HOUGH_GUIDED_WARP", False)
    monkeypatch.setattr(pipeline, "HOUGH_MIN_LENGTH_PX", 120)
    monkeypatch.setattr(pipeline, "HOUGH_SEARCH_BAND_PX", 45)

    height = 300
    width = 300
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[20:281, 30:271, :3] = 70
    rgba[20:281, 30:271, 3] = 255
    rgba[40:261, 48:53, :3] = 245
    rgba[40:261, 247:252, :3] = 245

    wall_poly = np.array(
        [[30.0, 280.0], [270.0, 280.0], [270.0, 20.0], [30.0, 20.0]],
        dtype=np.float64,
    )
    _result, info, _overlay, _warp_overlay, _bands = (
        pipeline._apply_group_hough_adjustment(
            ortho_rgba=rgba,
            wall_poly_px=wall_poly,
            rect_poly_px=wall_poly,
            per_building_out=tmp_path,
            geojson_base="building",
            facade_tag="group",
            edge_mask_override=None,
            allow_guided_warp=False,
        )
    )

    assert info["pipeline_stage"] == "before_post_rectification_sam"
    assert info["edge_source"] == "rectified_rgb_content_edges_before_sam"
    assert info["suppressed_projection_crop_boundary"] is True
    assert any(edge["selected_line"] is not None for edge in info["selected_edges"])


def test_group_hough_clips_warped_texture_and_overlay_to_wall_projection(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(pipeline, "ENABLE_ORTHO_HOUGH_DEBUG", True)
    monkeypatch.setattr(pipeline, "ENABLE_HOUGH_GUIDED_WARP", True)
    monkeypatch.setattr(pipeline, "SAVE_HOUGH_WARP_DEBUG", True)
    monkeypatch.setattr(pipeline, "HOUGH_SAVE_BAND_MASKS", False)
    monkeypatch.setattr(pipeline, "detect_hough_segments", lambda *_args, **_kwargs: [])

    def select_target_line(_lines, target_p0, target_p1, *_args, **_kwargs):
        return np.vstack([target_p0, target_p1]), {"accepted": True}

    monkeypatch.setattr(
        pipeline,
        "select_best_hough_line_for_target",
        select_target_line,
    )

    def fill_canvas_during_warp(ortho_rgba, **_kwargs):
        warped = np.full_like(ortho_rgba, 255)
        warped[:, :, :3] = (40, 80, 120)
        return warped

    monkeypatch.setattr(
        pipeline,
        "apply_hough_guided_ortho_warp",
        fill_canvas_during_warp,
    )

    captured = {}

    def capture_warp_overlay(img_pil, wall_quad_xy, out_path):
        captured["rgba"] = np.asarray(img_pil.convert("RGBA")).copy()
        Image.fromarray(captured["rgba"]).save(out_path)

    monkeypatch.setattr(pipeline, "save_hough_warp_overlay", capture_warp_overlay)

    height = 120
    width = 140
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[20:101, 25:116, :3] = 90
    rgba[20:101, 25:116, 3] = 255
    wall_poly = np.array(
        [[25.0, 100.0], [115.0, 100.0], [100.0, 20.0], [40.0, 20.0]],
        dtype=np.float64,
    )

    result, info, _overlay, warp_overlay, _bands = (
        pipeline._apply_group_hough_adjustment(
            ortho_rgba=rgba,
            wall_poly_px=wall_poly,
            rect_poly_px=wall_poly,
            per_building_out=tmp_path,
            geojson_base="building",
            facade_tag="group",
            edge_mask_override=None,
            allow_guided_warp=True,
        )
    )

    wall_mask = pipeline.build_wall_region_mask(height, width, wall_poly) > 0
    assert info["guided_warp_applied"] is True
    assert info["clipped_to_wall_projection_after_warp"] is True
    assert info["outside_wall_pixels_removed_after_warp"] > 0
    assert np.all(result[~wall_mask] == 0)
    assert np.all(captured["rgba"][~wall_mask] == 0)
    assert warp_overlay is not None
