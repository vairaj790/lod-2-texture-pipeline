import numpy as np
from PIL import Image

import lod2_texture_pipeline.projection as projection
import lod2_texture_pipeline.streetview as streetview
from lod2_texture_pipeline.streetview import (
    discover_recovery_panos_for_facade_group,
    facade_group_candidates_need_recovery,
    select_panos_for_facade_group,
)


def _facade_geometry():
    outline_xyz = np.array(
        [
            [-2.0, 10.0, 0.0],
            [2.0, 10.0, 0.0],
            [2.0, 10.0, 4.0],
            [-2.0, 10.0, 4.0],
        ],
        dtype=np.float64,
    )
    outline_m = np.array(
        [[-2.0, 0.0], [2.0, 0.0], [2.0, 4.0], [-2.0, 4.0]],
        dtype=np.float64,
    )
    origin = np.array([0.0, 10.0, 0.0], dtype=np.float64)

    def to_xyz(uv):
        values = np.asarray(uv, dtype=np.float64)
        if values.ndim == 1:
            return np.array([values[0], 10.0, values[1]], dtype=np.float64)
        return np.column_stack([
            values[:, 0],
            np.full(len(values), 10.0, dtype=np.float64),
            values[:, 1],
        ])

    frame = {
        "origin": origin,
        "u_dir": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "normal_xy": np.array([0.0, -1.0], dtype=np.float64),
        "to_xyz": to_xyz,
    }
    return {
        "frame": frame,
        "outline_xyz": outline_xyz,
        "outline_m": outline_m,
    }, outline_xyz


def test_native_source_selection_prefers_full_coverage(monkeypatch):
    geom, outline_xyz = _facade_geometry()
    candidates = [
        {"rec": {"pano_id": "near_partial", "utm": (0.0, 8.5)}, "u_clamped": 0.0},
        {"rec": {"pano_id": "far_complete", "utm": (0.0, 0.0)}, "u_clamped": 0.0},
    ]

    def fake_fetch(pano_id, heading, pitch, fov, api_key, size):
        color = (180, 80, 60) if pano_id == "near_partial" else (60, 120, 180)
        return Image.new("RGB", (640, 640), color), f"cache://{pano_id}", b"", "image/jpeg"

    fit_calls = []

    def fake_wireframe_fit(source, _outline_xyz, _facade_tag, _source_index):
        fit_calls.append(source["rec"]["pano_id"])
        source["wireframe_fit_H"] = np.eye(3, dtype=np.float64)
        source["wireframe_fit"] = {"applied": False, "reason": "test_identity"}
        source["wireframe_fit_overlay"] = None
        source["effective_camera_fit"] = {
            "attempted": False,
            "accepted": False,
            "reason": "test_identity",
        }

    monkeypatch.setattr(projection, "fetch_sv_image_by_id", fake_fetch)
    monkeypatch.setattr(projection, "_fit_raw_source_wireframe", fake_wireframe_fit)

    result = projection.select_facade_source_from_panos(
        geom=geom,
        pano_candidates=candidates,
        base_z=0.0,
        rect_xyz=outline_xyz,
        outline_xyz=outline_xyz,
        facade_tag="test_group",
        img_size="640x640",
    )

    assert result is not None
    assert result["image"].size == (640, 640)
    assert result["rec"]["pano_id"] == "far_complete"
    assert result["source_selection_ranking"][0]["full_frame_coverage"] is True
    assert result["source_selection_ranking"][0]["coverage_fraction"] == 1.0
    assert len(result["sources"]) == 2
    assert fit_calls == ["far_complete"]
    assert np.allclose(
        result["selected_source_raw_to_processing_image_H"],
        np.eye(3),
    )
    assert np.allclose(
        result["selected_source_corrected_to_aligned_image_H"],
        np.eye(3),
    )


def test_depth_global_mode_skips_wall_fit_for_every_candidate(monkeypatch):
    geom, outline_xyz = _facade_geometry()
    candidates = [
        {"rec": {"pano_id": "near_partial", "utm": (0.0, 8.5)}, "u_clamped": 0.0},
        {"rec": {"pano_id": "far_complete", "utm": (0.0, 0.0)}, "u_clamped": 0.0},
    ]
    fit_calls = []

    def fake_fetch(pano_id, heading, pitch, fov, api_key, size):
        return (
            Image.new("RGB", (640, 640), (80, 100, 120)),
            f"cache://{pano_id}",
            b"",
            "image/jpeg",
        )

    def unexpected_wireframe_fit(source, *_args, **_kwargs):
        fit_calls.append(source["rec"]["pano_id"])

    monkeypatch.setattr(projection, "fetch_sv_image_by_id", fake_fetch)
    monkeypatch.setattr(
        projection,
        "_fit_raw_source_wireframe",
        unexpected_wireframe_fit,
    )

    result = projection.select_facade_source_from_panos(
        geom=geom,
        pano_candidates=candidates,
        base_z=0.0,
        rect_xyz=outline_xyz,
        outline_xyz=outline_xyz,
        facade_tag="depth_global_test",
        img_size="640x640",
        facade_alignment_mode="depth_global",
    )

    assert result is not None
    assert fit_calls == []
    assert result["rec"]["pano_id"] == "far_complete"
    assert np.allclose(
        result["uv_outline"],
        result["uv_outline_before_wireframe_fit"],
    )
    selected = next(
        source for source in result["sources"]
        if source["selected_for_processing"]
    )
    assert selected["wireframe_fit"]["reason"] == "skipped_for_depth_global_alignment"
    unselected = next(
        source for source in result["sources"]
        if not source["selected_for_processing"]
    )
    assert unselected["wireframe_fit"]["reason"] == "not_selected_for_alignment"


def _install_screened_candidate_test_fetch(monkeypatch):
    def fake_fetch(pano_id, heading, pitch, fov, api_key, size):
        del heading, pitch, fov, api_key, size
        return (
            Image.new("RGB", (640, 640), (80, 100, 120)),
            f"cache://{pano_id}",
            b"",
            "image/jpeg",
        )

    monkeypatch.setattr(projection, "fetch_sv_image_by_id", fake_fetch)


def test_preselection_evaluates_all_candidates_and_prefers_osm_clear_view(monkeypatch):
    geom, outline_xyz = _facade_geometry()
    candidates = [
        {"rec": {"pano_id": "production_rank_one_blocked", "utm": (0.0, 0.0)}, "u_clamped": 0.0},
        {"rec": {"pano_id": "production_rank_two_clear", "utm": (1.0, 0.0)}, "u_clamped": 0.0},
    ]
    _install_screened_candidate_test_fetch(monkeypatch)
    evaluated = []

    def evaluate_all(sources):
        assert len(sources) == 2
        for source in sources:
            pano_id = source["rec"]["pano_id"]
            evaluated.append(pano_id)
            clear = pano_id.endswith("clear")
            source.update({
                "selection_projection_H": np.eye(3, dtype=np.float64),
                "depth_global_fit_evaluated_before_selection": True,
                "depth_global_fit_applied": True,
                "depth_global_fit_reason": "accepted",
                "depth_global_score_improvement": 0.25,
                "depth_global_fit_result": {
                    "homography": np.eye(3, dtype=np.float64),
                    "applied": True,
                    "reason": "accepted",
                },
                "external_building_occlusion_available": True,
                "external_building_occlusion_fraction": 0.0 if clear else 0.30,
                "external_building_clear": clear,
                "external_building_occlusion_mask": np.zeros((640, 640), dtype=bool),
                "external_building_candidate_blockers": [] if clear else ["osm_way_1"],
            })

    result = projection.select_facade_source_from_panos(
        geom=geom,
        pano_candidates=candidates,
        base_z=0.0,
        rect_xyz=outline_xyz,
        outline_xyz=outline_xyz,
        facade_tag="osm_clear_test",
        img_size="640x640",
        facade_alignment_mode="depth_global",
        candidate_preselection_evaluator=evaluate_all,
    )

    assert result is not None
    assert set(evaluated) == {
        "production_rank_one_blocked",
        "production_rank_two_clear",
    }
    assert result["rec"]["pano_id"] == "production_rank_two_clear"
    assert result["external_building_occlusion"]["clear"] is True
    assert result["selected_external_building_removal_mask"] is None
    assert all(
        row["depth_global_fit_evaluated_before_selection"]
        for row in result["source_selection_ranking"]
    )


def test_preselection_selects_least_blocked_view_and_returns_removal_mask(monkeypatch):
    geom, outline_xyz = _facade_geometry()
    candidates = [
        {"rec": {"pano_id": "more_blocked", "utm": (0.0, 0.0)}, "u_clamped": 0.0},
        {"rec": {"pano_id": "less_blocked", "utm": (1.0, 0.0)}, "u_clamped": 0.0},
    ]
    _install_screened_candidate_test_fetch(monkeypatch)
    expected_mask = np.zeros((640, 640), dtype=bool)
    expected_mask[240:320, 300:360] = True

    def evaluate_all(sources):
        for source in sources:
            less_blocked = source["rec"]["pano_id"] == "less_blocked"
            source.update({
                "selection_projection_H": np.eye(3, dtype=np.float64),
                "depth_global_fit_evaluated_before_selection": True,
                "depth_global_fit_applied": True,
                "depth_global_fit_reason": "accepted",
                "depth_global_score_improvement": 0.20,
                "depth_global_fit_result": {
                    "homography": np.eye(3, dtype=np.float64),
                    "applied": True,
                    "reason": "accepted",
                },
                "external_building_occlusion_available": True,
                "external_building_occlusion_fraction": 0.08 if less_blocked else 0.30,
                "external_building_clear": False,
                "external_building_occlusion_mask": (
                    expected_mask.copy()
                    if less_blocked else np.ones((640, 640), dtype=bool)
                ),
                "external_building_candidate_blockers": ["osm_way_1"],
            })

    result = projection.select_facade_source_from_panos(
        geom=geom,
        pano_candidates=candidates,
        base_z=0.0,
        rect_xyz=outline_xyz,
        outline_xyz=outline_xyz,
        facade_tag="osm_fallback_test",
        img_size="640x640",
        facade_alignment_mode="depth_global",
        candidate_preselection_evaluator=evaluate_all,
    )

    assert result is not None
    assert result["rec"]["pano_id"] == "less_blocked"
    assert result["external_building_occlusion"]["fallback_mask_required"] is True
    assert np.array_equal(
        result["selected_external_building_removal_mask"],
        expected_mask,
    )


def test_source_ranking_maximizes_net_visibility_instead_of_gating_on_osm_clear():
    _geom, outline_xyz = _facade_geometry()
    clear_partial = {
        "camera_xyz": np.array([0.0, 0.0, 2.5], dtype=np.float64),
        "target_model_visibility_available": True,
        "target_self_visibility_fraction": 1.0,
        "external_building_occlusion_available": True,
        "external_building_occlusion_fraction": 0.0,
        "external_building_clear": True,
    }
    slightly_blocked_complete = {
        "camera_xyz": np.array([1.0, 0.0, 2.5], dtype=np.float64),
        "target_model_visibility_available": True,
        "target_self_visibility_fraction": 0.9974,
        "external_building_occlusion_available": True,
        "external_building_occlusion_fraction": 0.0154,
        "external_building_clear": False,
    }
    base_metric = {
        "nondegenerate_projection": True,
        "projection_topology_valid": True,
        "inside_fraction": 1.0,
        "front_fraction": 1.0,
        "sane_span": True,
        "visible_projected_area": 40_000.0,
    }
    partial_metric = {
        **base_metric,
        "full_frame_coverage": False,
        "coverage_fraction": 0.161,
    }
    complete_metric = {
        **base_metric,
        "full_frame_coverage": True,
        "coverage_fraction": 1.0,
    }

    clear_terms = projection._target_visibility_selection_terms(
        clear_partial,
        partial_metric,
    )
    blocked_terms = projection._target_visibility_selection_terms(
        slightly_blocked_complete,
        complete_metric,
    )

    assert np.isclose(clear_terms["net_visibility_fraction"], 0.161)
    assert np.isclose(
        blocked_terms["net_visibility_fraction"],
        0.9974 * (1.0 - 0.0154),
    )
    assert projection._facade_source_selection_key(
        slightly_blocked_complete,
        complete_metric,
        outline_xyz,
    ) > projection._facade_source_selection_key(
        clear_partial,
        partial_metric,
        outline_xyz,
    )


def test_source_discovery_preserves_legacy_spacing_candidates():
    geom, _outline_xyz = _facade_geometry()
    pano_records = [
        {"pano_id": "p0", "utm": (-1.0, 0.0)},
        {"pano_id": "p1", "utm": (0.0, 0.0)},
        {"pano_id": "p2", "utm": (1.0, 0.0)},
    ]

    selected = select_panos_for_facade_group(
        geom,
        pano_records,
        max_panos=6,
        target_spacing_m=100.0,
    )

    assert [row["rec"]["pano_id"] for row in selected] == ["p0", "p2"]


def test_source_discovery_keeps_original_wall_prism_winner():
    geom, _outline_xyz = _facade_geometry()
    pano_records = [
        {"pano_id": "p0", "utm": (-1.0, 0.0)},
        {"pano_id": "p1", "utm": (0.0, 0.0)},
        {"pano_id": "p2", "utm": (1.0, 0.0)},
    ]
    wall_quad = np.array(
        [[-2.0, 10.0, 0.0], [2.0, 10.0, 0.0], [2.0, 10.0, 4.0], [-2.0, 10.0, 4.0]],
        dtype=np.float64,
    )

    selected = select_panos_for_facade_group(
        geom,
        pano_records,
        max_panos=6,
        target_spacing_m=100.0,
        wall_quads=[wall_quad],
        wall_normals=[np.array([0.0, -1.0, 0.0], dtype=np.float64)],
    )

    assert [row["rec"]["pano_id"] for row in selected] == ["p0", "p1", "p2"]
    p1 = next(row for row in selected if row["rec"]["pano_id"] == "p1")
    assert p1["legacy_wall_prism"] is True
    assert np.allclose(p1["legacy_wall_target_xyz"], [0.0, 10.0, 2.0])
    assert np.allclose(p1["legacy_wall_base_seg_xy"], [[-2.0, 10.0], [2.0, 10.0]])
    assert np.allclose(p1["legacy_wall_normal_xy"], [0.0, -1.0])


def test_legacy_policy_reproduces_original_wall_camera_and_selects_prism_winner(monkeypatch):
    geom, outline_xyz = _facade_geometry()
    wall_quad = outline_xyz.copy()
    pano_records = [
        {"pano_id": "coverage_candidate", "utm": (-1.0, 0.0)},
        {"pano_id": "legacy_winner", "utm": (0.0, 0.0)},
        {"pano_id": "other_candidate", "utm": (1.0, 0.0)},
    ]
    candidates = select_panos_for_facade_group(
        geom,
        pano_records,
        max_panos=6,
        target_spacing_m=100.0,
        wall_quads=[wall_quad],
        wall_normals=[np.array([0.0, -1.0, 0.0], dtype=np.float64)],
    )
    requests = {}

    def fake_fetch(pano_id, heading, pitch, fov, api_key, size):
        requests[pano_id] = (float(heading), float(pitch), float(fov))
        return Image.new("RGB", (640, 640), (80, 100, 120)), f"cache://{pano_id}", b"", "image/jpeg"

    def fake_wireframe_fit(source, _outline_xyz, _facade_tag, _source_index):
        source["wireframe_fit_H"] = np.eye(3, dtype=np.float64)
        source["wireframe_fit"] = {"applied": False, "reason": "test_identity"}
        source["wireframe_fit_overlay"] = None
        source["effective_camera_fit"] = {"attempted": False, "accepted": False}

    monkeypatch.setattr(projection, "fetch_sv_image_by_id", fake_fetch)
    monkeypatch.setattr(projection, "_fit_raw_source_wireframe", fake_wireframe_fit)

    result = projection.select_facade_source_from_panos(
        geom=geom,
        pano_candidates=candidates,
        base_z=0.0,
        rect_xyz=outline_xyz,
        outline_xyz=outline_xyz,
        facade_tag="legacy_test",
        img_size="640x640",
        source_selection_policy="legacy_wall_prism",
    )

    assert result is not None
    assert result["rec"]["pano_id"] == "legacy_winner"
    assert result["source_mode"] == "legacy_wall_prism_single_native_source"
    assert result["source_selection_policy"] == "legacy_wall_prism"
    heading, pitch, fov = requests["legacy_winner"]
    expected_fov = streetview.solve_fov_deg(
        np.array([0.0, 0.0]),
        0.0,
        (np.array([-2.0, 10.0]), np.array([2.0, 10.0])),
        np.array([0.0, -1.0]),
        buffer_m=projection.SIDE_BUFFER_M,
        safety_margin_deg=projection.FOV_MARGIN_DEG,
    )
    assert np.isclose(streetview.wrap_delta_deg(heading, 0.0), 0.0, atol=1.0e-5)
    assert np.isclose(pitch, np.degrees(np.arctan2(-0.5, 10.0)))
    assert np.isclose(fov, expected_fov)
    selected_source = next(source for source in result["sources"] if source["selected_for_processing"])
    assert selected_source["legacy_wall_framing"] is True


def test_outward_recovery_search_runs_only_for_nonfrontal_candidate_pool(monkeypatch):
    geom, _outline_xyz = _facade_geometry()
    fallback = [{
        "rec": {"pano_id": "tangent", "utm": (8.0, 10.0)},
        "forward_m": 0.0,
        "frontality": 0.0,
        "is_fallback": True,
    }]
    assert facade_group_candidates_need_recovery(fallback) is True

    class IdentityTransformer:
        @staticmethod
        def transform(x, y):
            return float(x), float(y)

    def fake_nearest(lat, lon, _api_key, radius, verbose):
        return {
            "status": "OK",
            "pano_id": f"recovered_{lon:.1f}",
            "location": {"lat": float(lat), "lng": float(lon)},
        }

    monkeypatch.setattr(streetview, "get_nearest_pano", fake_nearest)
    recovered = discover_recovery_panos_for_facade_group(
        geom,
        IdentityTransformer(),
        IdentityTransformer(),
        "test-key",
        existing_records=[fallback[0]["rec"]],
        forward_distances_m=(10.0,),
        lateral_pad_m=5.0,
        radius_m=3.0,
    )

    assert len(recovered) == 3
    selected = select_panos_for_facade_group(geom, recovered, max_panos=3)
    assert facade_group_candidates_need_recovery(selected) is False


def test_partial_source_ranking_prefers_coverage_before_resolution():
    _geom, outline_xyz = _facade_geometry()
    source = {"camera_xyz": np.array([0.0, 0.0, 2.5], dtype=np.float64)}
    higher_coverage = {
        "full_frame_coverage": False,
        "projection_topology_valid": True,
        "coverage_fraction": 0.82,
        "inside_fraction": 0.75,
        "front_fraction": 1.0,
        "sane_span": True,
        "visible_projected_area": 20_000.0,
    }
    higher_resolution = {
        **higher_coverage,
        "coverage_fraction": 0.61,
        "visible_projected_area": 120_000.0,
    }

    assert projection._facade_source_selection_key(
        source,
        higher_coverage,
        outline_xyz,
    ) > projection._facade_source_selection_key(
        source,
        higher_resolution,
        outline_xyz,
    )


def test_source_ranking_rejects_edge_on_full_coverage_projection():
    _geom, outline_xyz = _facade_geometry()
    source = {"camera_xyz": np.array([0.0, 0.0, 2.5], dtype=np.float64)}
    edge_on = {
        "nondegenerate_projection": False,
        "full_frame_coverage": True,
        "projection_topology_valid": True,
        "coverage_fraction": 1.0,
        "inside_fraction": 1.0,
        "front_fraction": 1.0,
        "sane_span": True,
        "visible_projected_area": 220.0,
    }
    usable_partial = {
        **edge_on,
        "nondegenerate_projection": True,
        "full_frame_coverage": False,
        "coverage_fraction": 0.75,
        "visible_projected_area": 40_000.0,
    }

    assert projection._facade_source_selection_key(
        source,
        usable_partial,
        outline_xyz,
    ) > projection._facade_source_selection_key(
        source,
        edge_on,
        outline_xyz,
    )


def test_rectification_orientation_keeps_roof_above_base():
    facade_metric = np.array(
        [[0.0, -4.0], [5.0, -4.0], [5.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    xmin, ymin = np.min(facade_metric, axis=0)
    xmax, ymax = np.max(facade_metric, axis=0)

    flip = projection.choose_orientation_from_poly(
        facade_metric,
        xmin,
        ymin,
        xmax,
        ymax,
        100.0,
    )
    transform = projection.S_meter_to_pixel(
        xmin,
        ymin,
        xmax,
        ymax,
        100.0,
        flip=flip,
    )
    facade_pixels = projection.apply_homography(facade_metric, transform)

    roof_y = np.mean(facade_pixels[2:, 1])
    base_y = np.mean(facade_pixels[:2, 1])
    assert flip is True
    assert roof_y < base_y
