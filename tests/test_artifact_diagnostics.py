import numpy as np
from PIL import Image

from lod2_texture_pipeline.pipeline import (
    _artifact_stage_label_and_rank,
    _include_artifact_in_contact_sheet,
    _mark_selected_candidate_overlay,
)


def test_selected_candidate_overlay_gets_check_badge(tmp_path):
    output_path = tmp_path / "building__source_pano02_overlay.png"
    Image.new("RGB", (320, 240), (30, 40, 50)).save(output_path)

    _mark_selected_candidate_overlay(output_path)

    marked = np.asarray(Image.open(output_path).convert("RGB"))
    badge = marked[:80, 240:]
    green = (
        (badge[:, :, 1] > 120)
        & (badge[:, :, 0] < 80)
        & (badge[:, :, 2] < 120)
    )
    white = np.all(badge > 235, axis=2)
    assert int(green.sum()) > 100
    assert int(white.sum()) > 20


def test_contact_sheet_includes_one_combined_card_per_candidate():
    assert not _include_artifact_in_contact_sheet(
        "building__source_pano02__prefit_semantic_guidance.png"
    )
    assert _include_artifact_in_contact_sheet(
        "building__source_pano02_overlay.png"
    )
    assert not _include_artifact_in_contact_sheet(
        "building__source_pano02.jpg"
    )
    assert not _include_artifact_in_contact_sheet(
        "building__source_pano02_target_model_visibility.png"
    )
    assert not _include_artifact_in_contact_sheet(
        "building__source_pano02_wireframe_fit.png"
    )
    assert not _include_artifact_in_contact_sheet(
        "sv__building__selected_native_source.jpg"
    )
    assert not _include_artifact_in_contact_sheet(
        "building__selected_depth_global_processing_overlay.png"
    )
    assert not _include_artifact_in_contact_sheet(
        "building__model_depth_mm_u16.png"
    )
    assert not _include_artifact_in_contact_sheet(
        "building__model_depth_boundary_candidate_mask.png"
    )
    assert _include_artifact_in_contact_sheet(
        "building__model_depth_visual.png"
    )
    assert _include_artifact_in_contact_sheet(
        "building__whole_model_depth_boundary_fit.png"
    )
    assert _include_artifact_in_contact_sheet(
        "building__model_depth_silhouette_mask.png"
    )

    visual_rank, _ = _artifact_stage_label_and_rank(
        "building__model_depth_visual.png"
    )
    boundary_rank, _ = _artifact_stage_label_and_rank(
        "building__whole_model_depth_boundary_fit.png"
    )
    silhouette_rank, _ = _artifact_stage_label_and_rank(
        "building__model_depth_silhouette_mask.png"
    )
    assert visual_rank < boundary_rank < silhouette_rank


def test_contact_sheet_orders_combined_candidate_cards():
    candidate_00_sam = _artifact_stage_label_and_rank(
        "building__source_pano00__prefit_semantic_guidance.png"
    )
    candidate_00_fit = _artifact_stage_label_and_rank(
        "building__source_pano00_overlay.png"
    )
    candidate_01_sam = _artifact_stage_label_and_rank(
        "building__source_pano01__prefit_semantic_guidance.png"
    )
    candidate_01_fit = _artifact_stage_label_and_rank(
        "building__source_pano01_overlay.png"
    )

    assert candidate_00_fit[0] < candidate_01_fit[0]
    assert candidate_00_sam[0] == candidate_00_fit[0]
    assert candidate_01_sam[0] == candidate_01_fit[0]
    assert "off contact sheet" in candidate_00_sam[1]
    assert (
        candidate_00_fit[1]
        == "01 candidate 00: SAM3-guided whole-model global fit"
    )


def test_contact_sheet_distinguishes_full_image_sam_from_actual_refit_evidence():
    osm_mask = _artifact_stage_label_and_rank(
        "building__selected_external_building_removal_mask.png"
    )
    osm_preview = _artifact_stage_label_and_rank(
        "building__selected_source_external_buildings_removed.png"
    )
    selected_sam = _artifact_stage_label_and_rank(
        "building__model_depth_prefit_semantic_guidance.png"
    )
    actual_evidence = _artifact_stage_label_and_rank(
        "building__selected_depth_global_fit_evidence_preview.png"
    )
    selected_refit = _artifact_stage_label_and_rank(
        "building__whole_model_depth_boundary_fit.png"
    )

    assert (
        osm_mask[0]
        < osm_preview[0]
        < selected_sam[0]
        < actual_evidence[0]
        < selected_refit[0]
    )
    assert "white pixels ignored by refit" in osm_mask[1]
    assert "preview only" in osm_preview[1]
    assert "full-image SAM3 guidance" in selected_sam[1]
    assert "ACTUAL refit evidence" in actual_evidence[1]
    assert "using 03e evidence" in selected_refit[1]


def test_contact_sheet_orders_hough_before_combined_sam_and_guarded_adjustment():
    hough_rank, _ = _artifact_stage_label_and_rank(
        "building__group__hough_overlay.png"
    )
    hough_warp_rank, _ = _artifact_stage_label_and_rank(
        "building__group__hough_warp_overlay.png"
    )
    sam_instances_rank, sam_instances_label = _artifact_stage_label_and_rank(
        "building__group__post_rectification_sam3_instances_overlay.png"
    )
    sam_refinement_rank, sam_refinement_label = _artifact_stage_label_and_rank(
        "building__group__post_rectification_sam3_overlay.png"
    )

    assert (
        hough_rank
        < hough_warp_rank
        < sam_instances_rank
        < sam_refinement_rank
    )
    assert "post-rectification cleanup SAM" in sam_instances_label
    assert "post-rectification cleanup SAM" in sam_refinement_label
    assert "SAM + guarded edge adjustment" in sam_refinement_label
