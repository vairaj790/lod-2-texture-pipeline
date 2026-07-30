import cv2
import numpy as np

import lod2_texture_pipeline.wireframe_fit as wireframe_fit
from lod2_texture_pipeline.wireframe_fit import (
    apply_homography,
    fit_wireframe_to_image,
    make_production_fit_config,
    transform_points_similarity,
)
from lod2_texture_pipeline.projection import project_outline_world_edges_near_clipped


def _test_config():
    return make_production_fit_config(
        coarse_scale_min=0.90,
        coarse_scale_max=1.10,
        coarse_scale_step=0.025,
        coarse_tx_min=-40.0,
        coarse_tx_max=40.0,
        coarse_tx_step=5.0,
        coarse_ty_min=-40.0,
        coarse_ty_max=40.0,
        coarse_ty_step=5.0,
        fine_tx_radius=5.0,
        fine_tx_step=1.0,
        fine_ty_radius=5.0,
        fine_ty_step=1.0,
        minimum_score_improvement=0.02,
    )


def _rectangle():
    return np.array(
        [[80.0, 80.0], [220.0, 80.0], [220.0, 210.0], [80.0, 210.0]],
        dtype=np.float64,
    )


def test_fit_recovers_one_shape_preserving_transform():
    original = _rectangle()
    center = original.mean(axis=0)
    expected = transform_points_similarity(original, 1.05, 0.0, 24.0, -17.0, center)
    image = np.zeros((300, 340, 3), dtype=np.uint8)
    cv2.polylines(image, [np.round(expected).astype(np.int32)], True, (255, 255, 255), 3)

    result = fit_wireframe_to_image(image, original, _test_config())

    assert result["applied"] is True
    assert np.mean(np.linalg.norm(result["fitted_points"] - expected, axis=1)) < 4.0
    assert np.allclose(
        apply_homography(original, result["homography"]),
        result["fitted_points"],
        atol=1e-6,
    )


def test_aligned_projection_is_not_moved():
    original = _rectangle()
    image = np.zeros((300, 340, 3), dtype=np.uint8)
    cv2.polylines(image, [original.astype(np.int32)], True, (255, 255, 255), 3)

    result = fit_wireframe_to_image(image, original, _test_config())

    assert result["applied"] is False
    assert np.allclose(result["fitted_points"], original)
    assert np.allclose(result["homography"], np.eye(3))


def test_slanted_invalid_side_crop_does_not_contribute_image_evidence():
    original = np.array(
        [[90.0, 35.0], [90.0, 165.0]],
        dtype=np.float64,
    )
    valid_line_image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(valid_line_image, (110, 35), (110, 165), (255, 255, 255), 3)
    obstructed_image = valid_line_image.copy()
    cv2.line(obstructed_image, (50, 20), (50, 180), (255, 255, 255), 7)
    excluded_evidence = np.zeros((200, 200), dtype=np.uint8)
    cv2.fillPoly(
        excluded_evidence,
        [np.array([[0, 0], [65, 0], [55, 199], [0, 199]], np.int32)],
        1,
    )
    excluded_evidence = excluded_evidence.astype(bool)
    valid_evidence = ~excluded_evidence
    config = make_production_fit_config(
        coarse_scale_min=1.0,
        coarse_scale_max=1.0,
        coarse_scale_step=0.1,
        coarse_tx_min=-40.0,
        coarse_tx_max=30.0,
        coarse_tx_step=5.0,
        coarse_ty_min=0.0,
        coarse_ty_max=0.0,
        coarse_ty_step=1.0,
        fine_scale_radius=0.0,
        fine_tx_radius=2.0,
        fine_tx_step=1.0,
        fine_ty_radius=0.0,
        fine_ty_step=1.0,
        minimum_score_improvement=0.01,
        minimum_mean_vertex_displacement_px=2.0,
    )

    clean_result = fit_wireframe_to_image(
        valid_line_image,
        original,
        config,
        segment_indices=[(0, 1)],
        valid_evidence_mask=valid_evidence,
    )
    obstructed_result = fit_wireframe_to_image(
        obstructed_image,
        original,
        config,
        segment_indices=[(0, 1)],
        valid_evidence_mask=valid_evidence,
    )

    assert obstructed_result["applied"] is True
    assert abs(float(obstructed_result["transform"]["tx"]) - 20.0) <= 3.0
    assert np.isclose(
        obstructed_result["transform"]["tx"],
        clean_result["transform"]["tx"],
    )
    assert not obstructed_result["canny"][excluded_evidence].any()
    assert not obstructed_result["line_map"][excluded_evidence].any()
    assert (
        obstructed_result["excluded_evidence_pixel_count"]
        == int(excluded_evidence.sum())
    )


def test_class_aware_semantic_boundary_breaks_ambiguous_edge_tie():
    original = np.array([[100.0, 35.0], [100.0, 165.0]], dtype=np.float64)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(image, (80, 35), (80, 165), (255, 255, 255), 3)
    cv2.line(image, (120, 35), (120, 165), (255, 255, 255), 3)
    roof_boundary = np.zeros((200, 200), dtype=bool)
    roof_boundary[:, 118:123] = True
    config = make_production_fit_config(
        coarse_scale_min=1.0,
        coarse_scale_max=1.0,
        coarse_scale_step=0.1,
        coarse_tx_min=-20.0,
        coarse_tx_max=20.0,
        coarse_tx_step=5.0,
        coarse_ty_min=0.0,
        coarse_ty_max=0.0,
        coarse_ty_step=1.0,
        fine_scale_radius=0.0,
        fine_tx_radius=0.0,
        fine_tx_step=1.0,
        fine_ty_radius=0.0,
        fine_ty_step=1.0,
        weight_semantic_boundary=3.0,
        minimum_score_improvement=0.01,
        minimum_mean_vertex_displacement_px=2.0,
    )

    result = fit_wireframe_to_image(
        image,
        original,
        config,
        segment_indices=[(0, 1)],
        segment_classes=["roof"],
        semantic_boundary_maps={"roof": roof_boundary},
    )

    assert result["applied"] is True
    assert abs(float(result["transform"]["tx"]) - 20.0) <= 6.0
    assert result["semantic_guidance_active"] is True
    assert result["semantic_boundary_score_after"] > 0.45
    assert (
        result["semantic_boundary_score_after"]
        > result["semantic_boundary_score_before"]
    )


def test_masked_fit_rejects_too_little_absolute_boundary_evidence():
    original = np.array([[100.0, 35.0], [100.0, 165.0]], dtype=np.float64)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(image, (120, 35), (120, 165), (255, 255, 255), 3)
    valid = np.zeros((200, 200), dtype=bool)
    valid[92:103, 98:123] = True
    config = make_production_fit_config(
        coarse_scale_min=1.0,
        coarse_scale_max=1.0,
        coarse_scale_step=0.1,
        coarse_tx_min=0.0,
        coarse_tx_max=20.0,
        coarse_tx_step=5.0,
        coarse_ty_min=0.0,
        coarse_ty_max=0.0,
        coarse_ty_step=1.0,
        fine_scale_radius=0.0,
        fine_tx_radius=0.0,
        fine_tx_step=1.0,
        fine_ty_radius=0.0,
        fine_ty_step=1.0,
        minimum_score_improvement=0.01,
        minimum_mean_vertex_displacement_px=1.0,
        minimum_masked_evidence_sample_count=8,
    )

    result = fit_wireframe_to_image(
        image,
        original,
        config,
        segment_indices=[(0, 1)],
        valid_evidence_mask=valid,
    )

    assert result["applied"] is False
    assert result["reason"] == "insufficient_unmasked_boundary_evidence"
    assert result["masked_evidence_gate_passed"] is False


def test_masked_search_prefers_admissible_evidence_count_over_higher_score(
    monkeypatch,
):
    original = np.array([[100.0, 100.0], [100.0, 200.0]], dtype=np.float64)
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    valid = np.ones((640, 640), dtype=bool)
    config = make_production_fit_config(
        coarse_scale_min=1.0,
        coarse_scale_max=1.0,
        coarse_scale_step=0.1,
        coarse_tx_min=0.0,
        coarse_tx_max=10.0,
        coarse_tx_step=5.0,
        coarse_ty_min=0.0,
        coarse_ty_max=0.0,
        coarse_ty_step=1.0,
        fine_scale_radius=0.0,
        fine_tx_radius=0.0,
        fine_tx_step=1.0,
        fine_ty_radius=0.0,
        fine_ty_step=1.0,
        minimum_score_improvement=0.01,
        minimum_mean_vertex_displacement_px=1.0,
        minimum_evidence_retention_ratio=0.8,
        minimum_masked_evidence_sample_count=8,
    )
    evaluated = []

    def fake_score_candidate(
        _candidate_points,
        _segment_indices,
        _scale,
        _rotation_deg,
        tx,
        _ty,
        *_args,
    ):
        if np.isclose(tx, 5.0):
            score, evidence_count = 1.30, 7
        elif np.isclose(tx, 10.0):
            score, evidence_count = 1.20, 8
        else:
            score, evidence_count = 1.00, 8
        evaluated.append((float(tx), score, evidence_count))
        return score, {
            "score": score,
            "edge_distance_score": 1.0,
            "long_line_score": 1.0,
            "semantic_boundary_score": 0.0,
            "semantic_evidence_sample_count": 0,
            "evidence_sample_count": evidence_count,
        }

    monkeypatch.setattr(
        wireframe_fit,
        "_score_candidate",
        fake_score_candidate,
    )
    result = fit_wireframe_to_image(
        image,
        original,
        config,
        segment_indices=[(0, 1)],
        valid_evidence_mask=valid,
    )

    assert any(np.isclose(tx, 5.0) and count == 7 for tx, _, count in evaluated)
    assert any(np.isclose(tx, 10.0) and count == 8 for tx, _, count in evaluated)
    assert result["applied"] is True
    assert np.isclose(result["transform"]["tx"], 10.0)
    assert result["transform"]["evidence_sample_count"] == 8


def test_semantic_map_on_zero_weight_segment_does_not_crash():
    points = np.array(
        [[30.0, 40.0], [130.0, 40.0], [30.0, 90.0], [130.0, 90.0]],
        dtype=np.float64,
    )
    image = np.zeros((150, 170, 3), dtype=np.uint8)
    cv2.line(image, (30, 40), (130, 40), (255, 255, 255), 2)
    roof_map = np.zeros((150, 170), dtype=bool)
    roof_map[90, 30:131] = True
    config = make_production_fit_config(
        coarse_scale_min=1.0,
        coarse_scale_max=1.0,
        coarse_scale_step=0.1,
        coarse_tx_min=0.0,
        coarse_tx_max=0.0,
        coarse_tx_step=1.0,
        coarse_ty_min=0.0,
        coarse_ty_max=0.0,
        coarse_ty_step=1.0,
        fine_scale_radius=0.0,
        fine_tx_radius=0.0,
        fine_tx_step=1.0,
        fine_ty_radius=0.0,
        fine_ty_step=1.0,
    )

    result = fit_wireframe_to_image(
        image,
        points,
        config,
        segment_indices=[(0, 1), (2, 3)],
        segment_weights=[1.0, 0.0],
        segment_classes=["wall", "roof"],
        semantic_boundary_maps={"roof": roof_map},
    )

    assert result["semantic_boundary_score_before"] == 0.0
    assert result["diagnostics_before"]["semantic_evidence_sample_count"] == 0


def test_near_plane_clipping_keeps_real_edges_disconnected():
    # The first edge is in front of the camera and the opposite edge is behind
    # it. A closed 2D projection would connect the two near-plane intersections
    # and create the apparent flip seen with very close Street View cameras.
    world_outline = np.array(
        [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    K = np.array(
        [[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    points, segments, clipped_world, info = project_outline_world_edges_near_clipped(
        world_outline,
        K,
        np.eye(3),
        np.zeros(3),
        near_m=0.75,
    )

    assert info["full_outline_topology_valid"] is False
    assert info["visible_real_edge_count"] == 3
    assert info["near_clipped_edge_count"] == 2
    assert segments == [(0, 1), (2, 3), (4, 5)]
    assert points.shape == (6, 2)
    assert clipped_world.shape == (6, 3)
    assert (3, 4) not in segments  # No artificial near-plane wrapper edge.
