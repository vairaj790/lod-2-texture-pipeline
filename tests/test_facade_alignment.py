import numpy as np
import pytest

from lod2_texture_pipeline import config
from lod2_texture_pipeline.facade_alignment import select_facade_alignment


def _quad(x0=0.0, y0=0.0, size=10.0):
    return np.array([
        [x0, y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0, y0 + size],
    ], dtype=np.float64)


def test_depth_global_is_the_default_alignment_mode():
    assert config.FACADE_ALIGNMENT_MODE == "depth_global"


def test_accepted_depth_global_fit_moves_raw_outline_and_rect_together():
    raw_outline = _quad()
    raw_rect = _quad(-2.0, -3.0, 14.0)
    wall_only_outline = raw_outline + np.array([100.0, 50.0])
    wall_only_rect = raw_rect + np.array([100.0, 50.0])
    H = np.array([
        [1.1, 0.0, 7.0],
        [0.0, 1.1, -4.0],
        [0.0, 0.0, 1.0],
    ])

    selected = select_facade_alignment(
        requested_mode="depth_global",
        wall_only_outline_px=wall_only_outline,
        wall_only_rect_px=wall_only_rect,
        raw_outline_px=raw_outline,
        raw_rect_px=raw_rect,
        depth_fit_result={"applied": True, "homography": H},
    )

    assert selected["effective_mode"] == "depth_global"
    assert not selected["fallback"]
    np.testing.assert_allclose(selected["outline_px"], raw_outline * 1.1 + [7.0, -4.0])
    np.testing.assert_allclose(selected["rect_px"], raw_rect * 1.1 + [7.0, -4.0])


def test_rejected_depth_fit_falls_back_to_wall_only():
    wall_outline = _quad(20.0, 30.0)
    wall_rect = _quad(18.0, 28.0, 14.0)
    selected = select_facade_alignment(
        requested_mode="depth_global",
        wall_only_outline_px=wall_outline,
        wall_only_rect_px=wall_rect,
        raw_outline_px=_quad(),
        raw_rect_px=_quad(),
        depth_fit_result={"applied": False, "reason": "insufficient_score_improvement"},
    )

    assert selected["effective_mode"] == "wall_only"
    assert selected["fallback"]
    assert "not_accepted" in selected["fallback_reason"]
    np.testing.assert_allclose(selected["outline_px"], wall_outline)
    np.testing.assert_allclose(selected["rect_px"], wall_rect)


def test_unknown_alignment_mode_is_rejected():
    with pytest.raises(ValueError, match="FACADE_ALIGNMENT_MODE"):
        select_facade_alignment(
            requested_mode="unknown",
            wall_only_outline_px=_quad(),
            wall_only_rect_px=_quad(),
            raw_outline_px=_quad(),
            raw_rect_px=_quad(),
            depth_fit_result=None,
        )
