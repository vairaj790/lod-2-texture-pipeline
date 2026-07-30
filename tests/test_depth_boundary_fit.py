import cv2
import numpy as np

from lod2_texture_pipeline.depth_boundary_fit import (
    create_depth_boundary_fit_overlay,
    create_depth_silhouette_shift_overlay,
    extract_depth_silhouette_geometry,
    filter_image_border_wrapper_segments,
    fit_depth_silhouette_to_image,
    project_semantic_model_boundary_edges,
)
from lod2_texture_pipeline.wireframe_fit import make_production_fit_config


def test_depth_silhouette_geometry_keeps_external_components_separate():
    depth = np.full((100, 140), np.nan, dtype=np.float32)
    depth[20:75, 15:70] = 12.0
    depth[35:80, 90:125] = 16.0

    geometry = extract_depth_silhouette_geometry(
        depth,
        minimum_area_px=100,
        minimum_component_fraction=0.01,
        contour_epsilon_px=0.5,
        maximum_points=40,
    )

    assert geometry["component_count"] == 2
    assert geometry["point_count"] >= 8
    assert len(geometry["segment_indices"]) == geometry["point_count"]
    ranges = geometry["contour_ranges"]
    for index0, index1 in geometry["segment_indices"]:
        assert any(start <= index0 < end and start <= index1 < end for start, end in ranges)


def test_depth_silhouette_geometry_removes_viewport_closure_before_simplifying():
    depth = np.full((100, 140), np.nan, dtype=np.float32)
    depth[20:, 30:] = 12.0

    geometry = extract_depth_silhouette_geometry(
        depth,
        minimum_area_px=100,
        contour_epsilon_px=2.0,
        maximum_points=40,
    )

    points = np.asarray(geometry["points"], dtype=np.float64)
    assert geometry["frame_wrappers_filtered"] is True
    assert geometry["image_border_wrapper_segment_count"] > 0
    assert geometry["boundary_run_count"] == 1
    assert len(geometry["segment_indices"]) == len(points) - 1

    def on_frame(point):
        return bool(
            point[0] == 0
            or point[0] == 139
            or point[1] == 0
            or point[1] == 99
        )

    for index0, index1 in geometry["segment_indices"]:
        assert not (on_frame(points[index0]) and on_frame(points[index1]))

    # The actual top and left silhouette remain; only the right/bottom
    # viewport closures are absent.
    assert np.any(np.isclose(points[:, 0], 30.0))
    assert np.any(np.isclose(points[:, 1], 20.0))


def test_legacy_frame_filter_removes_corner_bridge_but_keeps_x1_model_edge():
    points = np.array(
        [
            [139.0, 30.0],
            [70.0, 99.0],
            [1.0, 20.0],
            [1.0, 80.0],
        ],
        dtype=np.float64,
    )

    retained, excluded = filter_image_border_wrapper_segments(
        points,
        [(0, 1), (2, 3)],
        (100, 140),
        epsilon_px=0.5,
    )

    assert retained == [(2, 3)]
    assert excluded == [0]


def test_global_depth_fit_recovers_shift_without_changing_wall_local_fit():
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (55, 42), (115, 102), (255, 255, 255), 3)

    depth = np.full((160, 160), np.nan, dtype=np.float32)
    depth[50:111, 40:101] = 10.0
    raw_wall = np.array(
        [[45.0, 55.0], [95.0, 55.0], [95.0, 105.0], [45.0, 105.0]],
        dtype=np.float64,
    )
    wall_local = np.array(
        [[48.0, 53.0], [98.0, 53.0], [98.0, 103.0], [48.0, 103.0]],
        dtype=np.float64,
    )
    config = make_production_fit_config(
        allow_rotation=False,
        coarse_scale_min=1.0,
        coarse_scale_max=1.0,
        coarse_scale_step=0.1,
        coarse_tx_min=-25.0,
        coarse_tx_max=25.0,
        coarse_tx_step=5.0,
        coarse_ty_min=-20.0,
        coarse_ty_max=20.0,
        coarse_ty_step=4.0,
        fine_scale_radius=0.0,
        fine_tx_radius=3.0,
        fine_tx_step=1.0,
        fine_ty_radius=3.0,
        fine_ty_step=1.0,
        minimum_score_improvement=0.01,
        minimum_mean_vertex_displacement_px=2.0,
    )

    result = fit_depth_silhouette_to_image(
        image_bgr=image,
        full_model_depth=depth,
        raw_wall_outline_px=raw_wall,
        wall_local_fit_outline_px=wall_local,
        fit_config=config,
        minimum_area_px=100,
        contour_epsilon_px=0.5,
        maximum_points=40,
    )

    assert result["applied"] is True
    assert np.allclose(result["wall_local_fit_outline_px"], wall_local)
    transform = result["transform"]
    assert abs(float(transform["tx"]) - 15.0) <= 3.0
    assert abs(float(transform["ty"]) + 8.0) <= 3.0
    expected_depth_wall = raw_wall + np.array([15.0, -8.0])
    assert np.allclose(
        result["depth_global_fitted_wall_outline_px"],
        expected_depth_wall,
        atol=4.0,
    )

    overlay = create_depth_boundary_fit_overlay(
        image,
        result,
        config,
        line_thickness_px=2,
    )
    cyan = (
        (overlay[:, :, 0] > 180)
        & (overlay[:, :, 1] > 140)
        & (overlay[:, :, 2] < 100)
    )
    magenta = (
        (overlay[:, :, 0] > 180)
        & (overlay[:, :, 2] > 180)
        & (overlay[:, :, 1] < 100)
    )
    assert cyan.any()
    assert magenta.any()
    assert not magenta[75:95, 58:63].any()
    assert "candidate_depth_mask" not in result
    assert len(result["segment_classes"]) == len(result["segment_indices"])
    assert set(result["segment_classes"]) == {"silhouette"}

    silhouette_overlay = create_depth_silhouette_shift_overlay(
        result,
        config,
        line_thickness_px=2,
    )
    shifted_red = (
        (silhouette_overlay[:, :, 2] > 180)
        & (silhouette_overlay[:, :, 0] < 100)
        & (silhouette_overlay[:, :, 1] < 100)
    )
    assert shifted_red.any()
    assert np.any(shifted_red & ~result["raw_depth_mask"])


def test_projected_semantic_edges_keep_visible_external_model_boundaries():
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    depth[30:71, 30:71] = 10.0
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    model_edges = {
        "roof": np.array([[[-2.0, 2.0, 10.0], [2.0, 2.0, 10.0]]]),
        "wall": np.array([[[-2.0, -2.0, 10.0], [-2.0, 2.0, 10.0]]]),
        "base": np.array([[[-2.0, -2.0, 10.0], [2.0, -2.0, 10.0]]]),
    }

    geometry = project_semantic_model_boundary_edges(
        model_edges_xyz_by_class=model_edges,
        K=K,
        R_wc=np.eye(3),
        C=np.zeros(3),
        full_model_depth=depth,
        minimum_visible_run_px=5.0,
    )

    assert geometry["segment_classes"] == ["roof", "wall", "base"]
    assert geometry["visible_segment_counts"] == {
        "roof": 1,
        "wall": 1,
        "base": 1,
    }
    assert geometry["points"].shape == (6, 2)


def test_semantic_fit_prioritizes_roof_and_wall_over_misleading_base():
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    # Roof and side wall support the desired (+18, -12) shift. A long base
    # remains at its raw location and would otherwise attract many more samples.
    cv2.line(image, (48, 48), (138, 48), (255, 255, 255), 3)
    cv2.line(image, (48, 48), (48, 118), (255, 255, 255), 3)
    cv2.line(image, (15, 130), (165, 130), (255, 255, 255), 3)

    depth = np.full((180, 180), np.nan, dtype=np.float32)
    depth[60:131, 30:121] = 10.0
    semantic_geometry = {
        "points": np.array([
            [30.0, 60.0], [120.0, 60.0],
            [30.0, 60.0], [30.0, 130.0],
            [15.0, 130.0], [165.0, 130.0],
        ]),
        "segment_indices": [(0, 1), (2, 3), (4, 5)],
        "segment_classes": ["roof", "wall", "base"],
    }
    raw_wall = np.array(
        [[30.0, 60.0], [120.0, 60.0], [120.0, 130.0], [30.0, 130.0]],
        dtype=np.float64,
    )
    config = make_production_fit_config(
        coarse_scale_min=1.0,
        coarse_scale_max=1.0,
        coarse_scale_step=0.1,
        coarse_tx_min=-5.0,
        coarse_tx_max=25.0,
        coarse_tx_step=3.0,
        coarse_ty_min=-18.0,
        coarse_ty_max=3.0,
        coarse_ty_step=3.0,
        fine_scale_radius=0.0,
        fine_tx_radius=2.0,
        fine_tx_step=1.0,
        fine_ty_radius=2.0,
        fine_ty_step=1.0,
        minimum_score_improvement=0.01,
        minimum_mean_vertex_displacement_px=2.0,
    )

    result = fit_depth_silhouette_to_image(
        image_bgr=image,
        full_model_depth=depth,
        raw_wall_outline_px=raw_wall,
        wall_local_fit_outline_px=raw_wall,
        fit_config=config,
        minimum_area_px=100,
        semantic_boundary_geometry=semantic_geometry,
        semantic_class_weights={"roof": 3.0, "wall": 2.0, "base": 0.05},
    )

    assert result["fit_geometry_source"] == "visible_semantic_projected_edges"
    assert result["fit_segment_classes"] == ["roof", "wall", "base"]
    assert len(result["segment_classes"]) == len(result["segment_indices"])
    assert result["applied"] is True
    assert abs(float(result["transform"]["tx"]) - 18.0) <= 3.0
    assert abs(float(result["transform"]["ty"]) + 12.0) <= 3.0
