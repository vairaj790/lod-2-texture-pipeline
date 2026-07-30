from types import SimpleNamespace

import numpy as np

import lod2_texture_pipeline.projection as projection


def _quad_mesh(y, x_min=-2.0, x_max=2.0, z_min=0.0, z_max=4.0):
    return SimpleNamespace(
        vertices=np.array(
            [
                [x_min, y, z_min],
                [x_max, y, z_min],
                [x_max, y, z_max],
                [x_min, y, z_max],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def _forward_camera():
    return projection.build_pose_from_heading_pitch(
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        heading_deg=0.0,
        pitch_deg=0.0,
        img_size="200x200",
        fov_deg=90.0,
    )


def _selection_metric(area=20_000.0, coverage=1.0):
    return {
        "nondegenerate_projection": True,
        "projection_topology_valid": True,
        "full_frame_coverage": bool(coverage == 1.0),
        "coverage_fraction": float(coverage),
        "inside_fraction": float(coverage),
        "front_fraction": 1.0,
        "sane_span": True,
        "visible_projected_area": float(area),
    }


def test_target_wall_visibility_uses_complete_model_z_buffer(monkeypatch):
    monkeypatch.setattr(projection, "ENABLE_FACADE_SOURCE_MODEL_VISIBILITY", True)
    monkeypatch.setattr(projection, "FACADE_SOURCE_VISIBILITY_MASK_ERODE_PX", 1)
    target = _quad_mesh(y=10.0)
    nearer_wall = _quad_mesh(y=5.0, x_min=-1.0, x_max=1.0)
    K, R_wc, C = _forward_camera()

    visible = projection.evaluate_target_wall_model_visibility(
        [("target", target)],
        ["target"],
        K,
        R_wc,
        C,
        (200, 200),
    )
    occluded = projection.evaluate_target_wall_model_visibility(
        [("target", target), ("nearer", nearer_wall)],
        ["target"],
        K,
        R_wc,
        C,
        (200, 200),
    )

    assert visible["target_model_visibility_available"] is True
    assert visible["target_self_visibility_fraction"] > 0.999
    assert occluded["target_model_visibility_available"] is True
    assert occluded["target_self_visibility_fraction"] < 0.01
    assert occluded["target_occluded_pixel_count"] > 0


def test_source_ranking_prefers_visible_target_over_larger_projection():
    outline_xyz = _quad_mesh(y=10.0).vertices
    occluded_source = {
        "camera_xyz": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "target_model_visibility_available": True,
        "target_self_visibility_fraction": 0.12,
    }
    visible_source = {
        "camera_xyz": np.array([0.0, -8.0, 2.0], dtype=np.float64),
        "target_model_visibility_available": True,
        "target_self_visibility_fraction": 1.0,
    }

    occluded_key = projection._facade_source_selection_key(
        occluded_source,
        _selection_metric(area=55_000.0),
        outline_xyz,
        legacy_preference=True,
    )
    visible_key = projection._facade_source_selection_key(
        visible_source,
        _selection_metric(area=28_000.0),
        outline_xyz,
    )

    assert visible_key > occluded_key


def test_source_ranking_falls_back_to_highest_usable_target_visibility():
    outline_xyz = _quad_mesh(y=10.0).vertices
    source = {
        "camera_xyz": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "target_model_visibility_available": True,
    }
    more_usable = {
        **source,
        "target_self_visibility_fraction": 0.80,
    }
    less_usable = {
        **source,
        "target_self_visibility_fraction": 0.50,
    }

    assert projection._facade_source_selection_key(
        more_usable,
        _selection_metric(area=20_000.0, coverage=0.85),
        outline_xyz,
    ) > projection._facade_source_selection_key(
        less_usable,
        _selection_metric(area=90_000.0, coverage=1.0),
        outline_xyz,
    )
