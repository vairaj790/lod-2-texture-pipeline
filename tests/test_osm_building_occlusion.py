from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from experiments.osm_building_occlusion.run_experiment import (
    _prepare_contact_sheet_image,
)
from lod2_texture_pipeline.osm_occlusion import (
    OSMBuilding,
    build_osm_blocker_meshes,
    estimate_osm_building_height,
    evaluate_candidate_occlusion,
    remove_target_osm_buildings,
    select_candidate_with_osm_visibility,
)
import lod2_texture_pipeline.pipeline as pipeline
import lod2_texture_pipeline.projection as projection
from lod2_texture_pipeline.pipeline import _remove_external_building_pixels


def _target_quad(y=10.0):
    return np.array(
        [
            [-2.0, y, 0.0],
            [2.0, y, 0.0],
            [2.0, y, 4.0],
            [-2.0, y, 4.0],
        ],
        dtype=np.float64,
    )


def _quad_mesh(quad):
    return SimpleNamespace(
        vertices=np.asarray(quad, dtype=np.float64),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def _candidate(source_index=0, rank=0):
    return {
        "source_index": source_index,
        "source_selection_rank": rank,
        "camera_utm_xyz": [0.0, 0.0, 2.0],
        "projection_heading_deg": 0.0,
        "heading_deg": 0.0,
        "pitch_deg": 0.0,
        "fov_deg": 90.0,
        "target_usable_visibility_fraction": 1.0,
        "projected_coverage_fraction": 1.0,
    }


def test_osm_height_prefers_height_then_levels_and_min_level():
    height, minimum = estimate_osm_building_height(
        {"height": "40 ft", "building:levels": "99"}
    )
    assert np.isclose(height, 12.192)
    assert minimum == 0.0

    height, minimum = estimate_osm_building_height(
        {
            "building:levels": "4",
            "roof:levels": "1",
            "building:min_level": "1",
        }
    )
    assert height == 15.0
    assert minimum == 3.0


def test_target_osm_footprint_is_removed_but_neighbor_is_retained():
    target = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    matching = OSMBuilding(
        "way",
        1,
        Polygon([(0.2, 0.1), (10.1, 0.1), (10.1, 10.0), (0.2, 10.0)]),
        {},
        12.0,
    )
    neighbor = OSMBuilding(
        "way",
        2,
        Polygon([(12, 0), (20, 0), (20, 10), (12, 10)]),
        {},
        12.0,
    )

    blockers, excluded = remove_target_osm_buildings([matching, neighbor], target)

    assert [building.osm_id for building in blockers] == [2]
    assert excluded == ["way/1"]


def test_depth_comparison_detects_only_nearer_building_pixels():
    target_quad = _target_quad()
    target_meshes = [("target", _quad_mesh(target_quad))]
    blocker = OSMBuilding(
        "way",
        10,
        Polygon([(-0.4, 4.5), (0.4, 4.5), (0.4, 5.5), (-0.4, 5.5)]),
        {},
        4.0,
    )
    blocker_meshes, blocker_lookup = build_osm_blocker_meshes([blocker], ground_z=0.0)

    result = evaluate_candidate_occlusion(
        candidate=_candidate(),
        target_meshes=target_meshes,
        target_quads=[target_quad],
        blocker_meshes=blocker_meshes,
        blocker_lookup=blocker_lookup,
        image_size="200x200",
    )

    assert result["target_pixel_count"] > 0
    assert result["osm_occluded_pixel_count"] > 0
    assert 0.25 < result["osm_occluded_fraction"] < 0.75


def test_off_axis_osm_building_does_not_occlude_target_wall():
    target_quad = _target_quad()
    target_meshes = [("target", _quad_mesh(target_quad))]
    blocker = OSMBuilding(
        "way",
        11,
        Polygon([(6.0, 4.5), (8.0, 4.5), (8.0, 5.5), (6.0, 5.5)]),
        {},
        4.0,
    )
    blocker_meshes, blocker_lookup = build_osm_blocker_meshes([blocker], ground_z=0.0)

    result = evaluate_candidate_occlusion(
        candidate=_candidate(),
        target_meshes=target_meshes,
        target_quads=[target_quad],
        blocker_meshes=blocker_meshes,
        blocker_lookup=blocker_lookup,
        image_size="200x200",
    )

    assert result["osm_occluded_pixel_count"] == 0
    assert result["osm_occluded_fraction"] == 0.0


def test_occlusion_uses_depth_global_corrected_target_projection():
    target_quad = _target_quad()
    target_meshes = [("target", _quad_mesh(target_quad))]
    shift_x, shift_y = 18.0, -9.0
    correction = np.array(
        [
            [1.0, 0.0, shift_x],
            [0.0, 1.0, shift_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = evaluate_candidate_occlusion(
        candidate=_candidate(),
        target_meshes=target_meshes,
        target_quads=[target_quad],
        blocker_meshes=[],
        blocker_lookup={},
        image_size="200x200",
        target_alignment_H=correction,
    )

    raw_yx = np.mean(np.argwhere(result["raw_target_mask"]), axis=0)
    corrected_yx = np.mean(np.argwhere(result["target_mask"]), axis=0)
    displacement_yx = corrected_yx - raw_yx
    assert np.isclose(displacement_yx[1], shift_x, atol=1.0)
    assert np.isclose(displacement_yx[0], shift_y, atol=1.0)
    assert np.allclose(result["target_alignment_H"], correction)


def test_selection_maximizes_net_visibility_then_minimizes_obstruction():
    blocked = {
        "candidate": _candidate(source_index=0, rank=0),
        "target_pixel_count": 100,
        "osm_occluded_fraction": 0.25,
    }
    clear_rank_two = {
        "candidate": _candidate(source_index=2, rank=2),
        "target_pixel_count": 100,
        "osm_occluded_fraction": 0.0,
    }
    clear_rank_one = {
        "candidate": _candidate(source_index=1, rank=1),
        "target_pixel_count": 100,
        "osm_occluded_fraction": 0.004,
    }

    selection = select_candidate_with_osm_visibility(
        [blocked, clear_rank_two, clear_rank_one],
        clear_occlusion_fraction=0.005,
    )

    assert selection["selected"]["candidate"]["source_index"] == 2
    assert selection["fallback_mask_required"] is False


def test_selection_uses_least_occluded_view_and_requests_mask_when_all_blocked():
    more_blocked = {
        "candidate": _candidate(source_index=0, rank=0),
        "target_pixel_count": 100,
        "osm_occluded_fraction": 0.30,
    }
    less_blocked = {
        "candidate": _candidate(source_index=1, rank=1),
        "target_pixel_count": 100,
        "osm_occluded_fraction": 0.08,
    }

    selection = select_candidate_with_osm_visibility(
        [more_blocked, less_blocked],
        clear_occlusion_fraction=0.005,
    )

    assert selection["selected"]["candidate"]["source_index"] == 1
    assert selection["fallback_mask_required"] is True


def test_slightly_obstructed_complete_view_beats_clear_partial_view():
    clear_partial_candidate = _candidate(source_index=0, rank=0)
    clear_partial_candidate["target_usable_visibility_fraction"] = 0.161
    blocked_complete_candidate = _candidate(source_index=1, rank=1)
    blocked_complete_candidate["target_usable_visibility_fraction"] = 0.9974
    clear_partial = {
        "candidate": clear_partial_candidate,
        "target_pixel_count": 100,
        "osm_occluded_fraction": 0.0,
    }
    blocked_complete = {
        "candidate": blocked_complete_candidate,
        "target_pixel_count": 100,
        "osm_occluded_fraction": 0.0154,
    }

    selection = select_candidate_with_osm_visibility(
        [clear_partial, blocked_complete],
        clear_occlusion_fraction=0.005,
    )

    assert selection["selected"]["candidate"]["source_index"] == 1
    assert selection["fallback_mask_required"] is True
    assert (
        selection["selection_reason"]
        == "maximum_net_target_visibility_with_osm_removal"
    )


def test_close_camera_candidate_uses_clipped_projection_and_remains_eligible(monkeypatch):
    source = {
        "K": np.eye(3, dtype=np.float64),
        "Rwc": np.eye(3, dtype=np.float64),
        "C": np.zeros(3, dtype=np.float64),
        "img": Image.new("RGB", (64, 64), (80, 100, 120)),
    }
    outline = np.array(
        [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, 2.0],
            [-1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )

    monkeypatch.setattr(
        pipeline,
        "project_outline_world_edges_near_clipped",
        lambda *_args, **_kwargs: (
            np.zeros((0, 2), dtype=np.float64),
            [],
            np.zeros((0, 3), dtype=np.float64),
            {"full_outline_topology_valid": False},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_render_model_depth_view",
        lambda **_kwargs: np.ones((64, 64), dtype=np.float32),
    )
    clipped_outline = np.array(
        [[8.0, 8.0], [56.0, 8.0], [56.0, 56.0], [8.0, 56.0]],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        pipeline,
        "project_polygon_world_to_image_clipped",
        lambda *_args, **_kwargs: clipped_outline.copy(),
    )
    fit_inputs = []

    def fake_depth_fit(**kwargs):
        fit_inputs.append({
            "raw_wall_outline_px": np.asarray(
                kwargs["raw_wall_outline_px"]
            ).copy(),
            "valid_image_evidence_mask": np.asarray(
                kwargs["valid_image_evidence_mask"],
                dtype=bool,
            ).copy(),
            "semantic_image_boundary_maps": dict(
                kwargs["semantic_image_boundary_maps"] or {}
            ),
            "semantic_image_guidance_metadata": dict(
                kwargs["semantic_image_guidance_metadata"]
            ),
        })
        return {
            "applied": True,
            "homography": np.eye(3, dtype=np.float64),
            "reason": "accepted",
            "score_improvement": 0.25,
        }

    monkeypatch.setattr(
        pipeline,
        "fit_depth_silhouette_to_image",
        fake_depth_fit,
    )
    monkeypatch.setattr(
        pipeline,
        "MODEL_DEPTH_BOUNDARY_USE_SEMANTIC_GUIDES",
        False,
    )
    monkeypatch.setattr(
        pipeline,
        "ENABLE_MODEL_DEPTH_PREFIT_SEMANTIC_GUIDANCE",
        True,
    )

    pipeline._candidate_depth_global_and_osm_preselection(
        [source],
        outline_xyz=outline,
        target_meshes=[],
        target_quads=[],
        meshes_named=[],
        model_boundary_edges_xyz={},
        osm_context={
            "available": False,
            "reason": "test",
            "blocker_meshes": [],
            "blocker_lookup": {},
        },
        facade_tag="close_camera_test",
    )

    assert source["depth_global_fit_applied"] is True
    assert source["depth_global_fit_reason"] == "accepted"
    assert source["wireframe_projection_info"]["full_outline_topology_valid"] is False
    assert source["selection_uses_near_clipped_projection"] is True
    assert source["selection_projection_topology_valid"] is True
    assert np.array_equal(
        source["selection_visible_wall_outline_px"],
        clipped_outline,
    )
    assert len(fit_inputs) == 1
    assert np.array_equal(
        fit_inputs[0]["raw_wall_outline_px"],
        clipped_outline,
    )
    assert fit_inputs[0]["valid_image_evidence_mask"].shape == (64, 64)
    assert fit_inputs[0]["valid_image_evidence_mask"].dtype == bool
    assert set(fit_inputs[0]["semantic_image_boundary_maps"]) == {
        "roof",
        "wall",
        "base",
        "silhouette",
    }
    assert not any(
        mask.any()
        for mask in fit_inputs[0]["semantic_image_boundary_maps"].values()
    )
    assert (
        fit_inputs[0]["semantic_image_guidance_metadata"]["stage"]
        == "candidate_preselection_before_global_depth_fit"
    )
    assert source["external_building_occlusion_available"] is False
    metric = projection._source_projection_metric(
        source,
        outline,
        (64, 64),
    )
    assert metric["projection_topology_valid"] is True
    assert metric["nondegenerate_projection"] is True
    assert metric["uses_near_plane_clipped_projection"] is True


def test_candidate_overlay_draws_raw_and_fitted_whole_model_not_wall_edges(
    tmp_path,
):
    raw_model = np.array(
        [[20.0, 80.0], [80.0, 80.0], [80.0, 150.0], [20.0, 150.0]],
        dtype=np.float64,
    )
    fitted_model = raw_model + np.array([30.0, 0.0])
    legacy_wall = np.array(
        [[140.0, 90.0], [160.0, 90.0], [160.0, 150.0], [140.0, 150.0]],
        dtype=np.float64,
    )
    source = {
        "img": Image.new("RGB", (180, 180), (255, 255, 255)),
        "selection_visible_wall_outline_px": legacy_wall,
        "selection_projection_H": np.eye(3, dtype=np.float64),
        "selection_projection_topology_valid": True,
        "selection_real_wall_edge_points_px": legacy_wall,
        "selection_real_wall_edge_segments": [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ],
        "depth_global_fit_result": {
            "applied": True,
            "original_points": raw_model,
            "candidate_points": fitted_model,
            "fitted_points": fitted_model,
            "segment_indices": [(0, 1), (1, 2), (2, 3), (3, 0)],
        },
        "depth_global_fit_applied": True,
        "external_building_occlusion_available": False,
    }
    output_path = tmp_path / "candidate_overlay.png"

    pipeline._save_candidate_projection_screening_overlay(
        source,
        legacy_wall,
        output_path,
    )

    rendered = np.asarray(Image.open(output_path).convert("RGB"))
    cyan = (
        (rendered[:, :, 0] < 80)
        & (rendered[:, :, 1] > 150)
        & (rendered[:, :, 2] > 180)
    )
    magenta = (
        (rendered[:, :, 0] > 150)
        & (rendered[:, :, 1] < 80)
        & (rendered[:, :, 2] > 150)
    )
    assert cyan[95:140, 17:24].any()
    assert magenta[95:140, 107:114].any()
    assert np.all(rendered[120, 145] > 245)
    assert np.all(rendered[120, 65] > 245)


def test_candidate_overlay_never_displays_rejected_whole_model_proposal(
    tmp_path,
):
    raw_model = np.array(
        [[20.0, 80.0], [70.0, 80.0], [70.0, 140.0], [20.0, 140.0]],
        dtype=np.float64,
    )
    rejected_candidate = raw_model + np.array([70.0, 0.0])
    source = {
        "img": Image.new("RGB", (180, 180), (255, 255, 255)),
        "selection_projection_H": np.eye(3, dtype=np.float64),
        "depth_global_fit_result": {
            "applied": False,
            "reason": "insufficient_score_improvement",
            "original_points": raw_model,
            "candidate_points": rejected_candidate,
            "fitted_points": raw_model.copy(),
            "segment_indices": [(0, 1), (1, 2), (2, 3), (3, 0)],
        },
        "depth_global_fit_applied": False,
        "external_building_occlusion_available": False,
    }
    output_path = tmp_path / "candidate_rejected_overlay.png"

    pipeline._save_candidate_projection_screening_overlay(
        source,
        np.zeros((0, 2), dtype=np.float64),
        output_path,
    )

    rendered = np.asarray(Image.open(output_path).convert("RGB"))
    cyan = (
        (rendered[:, :, 0] < 80)
        & (rendered[:, :, 1] > 150)
        & (rendered[:, :, 2] > 180)
    )
    assert cyan[95:130, 17:24].any()
    assert np.array_equal(rendered[110, 140], [255, 255, 255])


def test_candidate_overlay_combines_sam_guidance_with_real_whole_model_edges(
    tmp_path,
):
    shape = (180, 180)
    raw_model = np.array(
        [[25.0, 85.0], [75.0, 65.0], [110.0, 90.0], [105.0, 150.0]],
        dtype=np.float64,
    )
    fitted_model = raw_model + np.array([18.0, -3.0])
    target = np.zeros(shape, dtype=bool)
    target[105:135, 125:160] = True
    excluded = np.zeros(shape, dtype=bool)
    excluded[140:165, 125:150] = True
    roof_guide = np.zeros(shape, dtype=bool)
    roof_guide[103, 125:160] = True
    wall_guide = np.zeros(shape, dtype=bool)
    wall_guide[105:135, 160] = True
    source = {
        "img": Image.new("RGB", shape[::-1], (255, 255, 255)),
        "selection_projection_H": np.array(
            [[1.0, 0.0, 18.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "depth_global_prefit_semantic_guidance": {
            "raw_projection_mask": target,
            "local_search_mask": np.ones(shape, dtype=bool),
            "selected_building_mask": target,
            "excluded_evidence_mask": excluded,
            "boundary_maps": {
                "roof": roof_guide,
                "wall": wall_guide,
            },
        },
        "depth_global_fit_result": {
            "applied": True,
            "fit_geometry_source": "visible_semantic_projected_edges",
            "fit_original_points": raw_model,
            "fit_fitted_points": fitted_model,
            "fit_segment_indices": [(0, 1), (1, 2), (2, 3)],
            # A deliberately unrelated diagnostic silhouette must not be used
            # when real projected whole-model edges are available.
            "original_points": np.array(
                [[5.0, 160.0], [175.0, 160.0], [175.0, 179.0]],
                dtype=np.float64,
            ),
            "fitted_points": np.array(
                [[5.0, 150.0], [175.0, 150.0], [175.0, 169.0]],
                dtype=np.float64,
            ),
            "segment_indices": [(0, 1), (1, 2)],
            "transform": {
                "scale": 1.0,
                "tx": 18.0,
                "ty": -3.0,
            },
            "score_improvement": 0.12,
        },
        "depth_global_fit_applied": True,
        "external_building_occlusion_available": False,
    }
    output_path = tmp_path / "combined_candidate_overlay.png"

    pipeline._save_candidate_projection_screening_overlay(
        source,
        np.zeros((0, 2), dtype=np.float64),
        output_path,
    )

    rendered = np.asarray(Image.open(output_path).convert("RGB"))
    cyan_line = (
        (rendered[:, :, 0] < 80)
        & (rendered[:, :, 1] > 150)
        & (rendered[:, :, 2] > 180)
    )
    magenta_line = (
        (rendered[:, :, 0] > 150)
        & (rendered[:, :, 1] < 80)
        & (rendered[:, :, 2] > 150)
    )
    pink_fill = (
        (rendered[:, :, 0] > 240)
        & (rendered[:, :, 1] < 220)
        & (rendered[:, :, 2] > 210)
    )
    green_guide = (
        (rendered[:, :, 0] < 40)
        & (rendered[:, :, 1] > 200)
        & (rendered[:, :, 2] < 130)
    )
    assert cyan_line[80:145, 20:112].any()
    assert magenta_line[75:145, 38:130].any()
    assert pink_fill[145:160, 130:145].any()
    assert green_guide[110:130, 155:165].any()
    assert source["depth_global_projection_overlay_geometry"] == (
        "visible_real_whole_model_edges"
    )
    # The unrelated bottom diagnostic silhouette was not drawn.
    assert not cyan_line[157:164, 40:160].any()


def test_candidate_overlay_never_moves_viewport_closure_into_image(tmp_path):
    raw_model = np.array(
        [
            [20.0, 40.0],
            [100.0, 20.0],
            [179.0, 50.0],
            [179.0, 179.0],
            [50.0, 179.0],
            [20.0, 120.0],
        ],
        dtype=np.float64,
    )
    fitted_model = raw_model + np.array([-20.0, -20.0])
    source = {
        "img": Image.new("RGB", (180, 180), (255, 255, 255)),
        "selection_projection_H": np.array(
            [[1.0, 0.0, -20.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "depth_global_fit_result": {
            "applied": True,
            "original_points": raw_model,
            "fitted_points": fitted_model,
            "segment_indices": [
                (0, 1),
                (1, 2),
                (2, 3),  # synthetic right-frame closure
                (3, 4),  # synthetic bottom-frame closure
                (4, 5),
                (5, 0),
            ],
        },
        "depth_global_fit_applied": True,
        "external_building_occlusion_available": False,
    }
    output_path = tmp_path / "frame_safe_candidate_overlay.png"

    pipeline._save_candidate_projection_screening_overlay(
        source,
        np.zeros((0, 2), dtype=np.float64),
        output_path,
    )

    rendered = np.asarray(Image.open(output_path).convert("RGB"))
    magenta = (
        (rendered[:, :, 0] > 150)
        & (rendered[:, :, 1] < 80)
        & (rendered[:, :, 2] > 150)
    )
    assert magenta.any()
    assert not magenta[55:150, 157:162].any()
    assert not magenta[157:162, 45:150].any()
    assert (
        source["depth_global_projection_overlay_excluded_frame_segment_count"]
        == 2
    )


def test_contact_sheet_preview_makes_transparent_removal_visible():
    pixels = np.full((28, 28, 4), (20, 40, 60, 255), dtype=np.uint8)
    pixels[7:21, 7:21, 3] = 0

    preview = _prepare_contact_sheet_image(Image.fromarray(pixels, mode="RGBA"))

    assert preview.mode == "RGB"
    assert preview.getpixel((2, 2)) == (20, 40, 60)
    assert preview.getpixel((10, 10)) in {(238, 238, 238), (213, 213, 213)}


def test_external_obstruction_is_removed_only_inside_selected_wall_alpha():
    wall = np.zeros((12, 12), dtype=bool)
    wall[2:10, 2:10] = True
    obstruction = np.zeros_like(wall)
    obstruction[5:11, 5:11] = True

    retained, removed = _remove_external_building_pixels(wall, obstruction)

    assert np.array_equal(removed, wall & obstruction)
    assert np.array_equal(retained, wall & ~obstruction)
    assert not retained[5:10, 5:10].any()
    assert retained[2:5, 2:5].all()


def test_osm_obstruction_uses_slanted_lr_style_side_crop_for_refit():
    target = np.zeros((100, 120), dtype=np.uint8)
    cv2.fillPoly(
        target,
        [np.array([[30, 10], [90, 10], [80, 90], [20, 90]], np.int32)],
        1,
    )
    local_obstruction = np.zeros_like(target)
    cv2.fillPoly(
        local_obstruction,
        [np.array([[30, 10], [56, 10], [46, 90], [20, 90]], np.int32)],
        1,
    )
    local_obstruction = local_obstruction.astype(bool) & target.astype(bool)

    exclusion, info = pipeline._external_building_lr_side_exclusion_mask(
        local_obstruction,
        target.astype(bool),
    )

    assert info["applied"] is True
    assert info["reason"] == "obstruction_side_half_plane_extended_top_to_bottom"
    assert exclusion[:, 0].all()
    assert not exclusion[:, -1].any()
    top_cut = int(np.flatnonzero(~exclusion[0])[0])
    bottom_cut = int(np.flatnonzero(~exclusion[-1])[0])
    assert abs(top_cut - bottom_cut) >= 5
    assert (exclusion & local_obstruction).sum() == local_obstruction.sum()
    assert (target.astype(bool) & ~exclusion).any()


def test_rejected_selected_osm_refit_keeps_unshifted_raw_depth_projection():
    original_points = np.array(
        [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]],
        dtype=np.float64,
    )
    numerical_candidate = original_points + np.array([3.0, -2.0])
    raw_wall = original_points.copy()
    exclusion = np.zeros((100, 100), dtype=bool)
    exclusion[:, 35:45] = True
    result = {
        "applied": False,
        "reason": "insufficient_score_improvement",
        "homography": np.eye(3, dtype=np.float64),
        "candidate_homography": np.array(
            [[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "original_points": original_points,
        "candidate_points": numerical_candidate,
        "fitted_points": original_points.copy(),
        "depth_global_fitted_wall_outline_px": raw_wall.copy(),
        "score_before": 2.0,
        "score_after": 2.01,
        "score_improvement": 0.01,
        "mean_vertex_displacement_px": 3.6,
        "diagnostics_before": {
            "evidence_sample_count": 20,
            "semantic_boundary_score": 0.40,
            "edge_distance_score": 0.55,
        },
        "semantic_boundary_score_before": 0.40,
        "semantic_boundary_score_after": 0.62,
        "transform": {
            "scale": 1.0,
            "rotation_deg": 0.0,
            "tx": 3.0,
            "ty": -2.0,
            "score": 2.01,
            "evidence_sample_count": 20,
            "semantic_boundary_score": 0.62,
            "edge_distance_score": 0.70,
            "transform_center_x": 50.0,
            "transform_center_y": 50.0,
        },
    }

    selected = pipeline._finalize_selected_osm_masked_depth_refit(
        result,
        raw_wall_outline_px=raw_wall,
        exclusion_mask=exclusion,
    )

    assert selected["applied"] is True
    assert selected["selected_source_osm_refit"] is True
    assert selected["selected_source_osm_refit_numerical_fit_applied"] is False
    assert selected["selected_source_osm_refit_identity_fallback"] is True
    assert selected["reason"] == "selected_osm_refit_kept_unshifted_raw_projection"
    assert np.array_equal(selected["homography"], np.eye(3))
    assert np.array_equal(selected["fitted_points"], original_points)
    assert np.array_equal(
        selected["depth_global_fitted_wall_outline_px"],
        raw_wall,
    )
    assert selected["excluded_image_evidence_column_count"] == 10
    assert selected["transform"]["semantic_boundary_score"] == 0.40
    assert selected["semantic_boundary_score_after"] == 0.40
    assert (
        selected[
            "selected_source_osm_refit_numerical_semantic_boundary_score_after"
        ]
        == 0.62
    )


def test_zero_evidence_osm_refit_does_not_restore_unsafe_preselection_shift():
    original_points = np.array(
        [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]],
        dtype=np.float64,
    )
    exclusion = np.ones((100, 100), dtype=bool)
    result = {
        "applied": False,
        "reason": "insufficient_unmasked_boundary_evidence",
        "homography": np.eye(3, dtype=np.float64),
        "candidate_homography": np.eye(3, dtype=np.float64),
        "original_points": original_points,
        "candidate_points": original_points.copy(),
        "fitted_points": original_points.copy(),
        "depth_global_fitted_wall_outline_px": original_points.copy(),
        "score_before": -1.0e9,
        "score_after": -1.0e9,
        "score_improvement": 0.0,
        "mean_vertex_displacement_px": 0.0,
        "diagnostics_before": {"evidence_sample_count": 0},
        "transform": {
            "scale": 1.0,
            "rotation_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "score": -1.0e9,
            "evidence_sample_count": 0,
            "transform_center_x": 50.0,
            "transform_center_y": 50.0,
        },
    }

    selected = pipeline._finalize_selected_osm_masked_depth_refit(
        result,
        raw_wall_outline_px=original_points,
        exclusion_mask=exclusion,
    )

    assert selected["applied"] is True
    assert (
        selected["reason"]
        == "selected_osm_refit_kept_unshifted_raw_projection_no_evidence"
    )
    assert np.array_equal(selected["homography"], np.eye(3))
    assert np.array_equal(
        selected["depth_global_fitted_wall_outline_px"],
        original_points,
    )


def test_osm_finalizer_unions_explicit_and_semantic_exclusions():
    points = np.array(
        [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
        dtype=np.float64,
    )
    exclusion = np.zeros((50, 50), dtype=bool)
    exclusion[:, :5] = True
    semantic_valid = np.ones((50, 50), dtype=bool)
    semantic_valid[:, 45:] = False
    result = {
        "applied": True,
        "reason": "accepted_score_improvement",
        "homography": np.eye(3, dtype=np.float64),
        "original_points": points,
        "fitted_points": points.copy(),
        "depth_global_fitted_wall_outline_px": points.copy(),
        "diagnostics_before": {"evidence_sample_count": 20},
        "transform": {
            "scale": 1.0,
            "rotation_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "evidence_sample_count": 20,
        },
    }

    selected = pipeline._finalize_selected_osm_masked_depth_refit(
        result,
        raw_wall_outline_px=points,
        exclusion_mask=exclusion,
        valid_evidence_mask=semantic_valid,
    )

    assert selected["applied"] is True
    assert selected["selected_source_osm_refit_identity_fallback"] is False
    assert selected["osm_excluded_image_evidence_pixel_count"] == 250
    assert (
        selected["semantic_or_locality_excluded_image_evidence_pixel_count"]
        == 250
    )
    assert selected["excluded_image_evidence_pixel_count"] == 500
    assert selected["valid_image_evidence_pixel_count"] == 2000
