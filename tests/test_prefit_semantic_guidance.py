import json

import numpy as np
import pytest
from PIL import Image

from lod2_texture_pipeline import pipeline
from lod2_texture_pipeline.prefit_semantic_guidance import (
    PrefitSemanticGuidanceConfig,
    build_prefit_semantic_guidance,
    create_prefit_semantic_guidance_overlay,
)


def _config(**overrides):
    values = {
        "search_dilation_px": 14,
        "target_association_distance_px": 10.0,
        "target_min_overlap_pixels": 5,
        "target_min_overlap_fraction": 0.01,
        "target_min_local_fraction": 0.20,
        "target_relative_score_threshold": 0.35,
        "target_min_new_projection_pixels": 5,
        "occluder_dilation_px": 2,
        "context_adjacency_px": 2,
        "envelope_tolerance_px": 1,
        "boundary_thickness_px": 1,
    }
    values.update(overrides)
    return PrefitSemanticGuidanceConfig(**values)


def test_guidance_selects_projection_local_target_and_builds_class_boundaries():
    height, width = 100, 120
    projection = np.zeros((height, width), dtype=bool)
    projection[30:80, 30:90] = True

    target_building = np.zeros_like(projection)
    target_building[32:78, 32:88] = True
    distant_building = np.zeros_like(projection)
    distant_building[5:24, 96:116] = True

    target_roof = np.zeros_like(projection)
    target_roof[28:36, 32:88] = True
    distant_roof = np.zeros_like(projection)
    distant_roof[2:8, 98:116] = True

    sky = np.zeros_like(projection)
    sky[:29, :] = True
    ground = np.zeros_like(projection)
    ground[78:, :] = True
    tree = np.zeros_like(projection)
    tree[38:67, 84:94] = True
    road_stack = ground[None, :, :]

    result = build_prefit_semantic_guidance(
        {
            "building": np.stack([target_building, distant_building]),
            "roof": np.stack([target_roof, distant_roof]),
            "clear sky": sky,
            "road": road_stack,
            "tree": tree,
        },
        projection,
        config=_config(),
    )

    assert road_stack.shape == (1, height, width)
    assert result["selected_building_mask"][50, 40]
    assert not result["selected_building_mask"][10, 105]
    assert result["selected_roof_mask"][30, 50]
    assert not result["selected_roof_mask"][4, 105]
    assert result["metadata"]["roles"]["building"]["selected_instances"] == 1
    assert result["metadata"]["roles"]["roof"]["selected_instances"] == 1
    assert (
        result["metadata"]["selected_target_instances"]["building"][0][
            "overlap_px"
        ]
        > 0
    )

    assert result["local_search_mask"][25, 30]
    assert not result["local_search_mask"][0, 0]
    assert result["valid_evidence_mask"][50, 40]
    assert not result["valid_evidence_mask"][50, 88]
    assert result["excluded_evidence_mask"][50, 82]

    boundaries = result["boundary_maps"]
    assert boundaries["roof"][28, 50]
    assert boundaries["base"][77, 50]
    assert boundaries["wall"][50, 32]
    assert not boundaries["wall"][50, 87]
    assert boundaries["silhouette"].any()
    assert set(result["metadata"]["boundary_classes"]) == {
        "roof",
        "wall",
        "base",
        "silhouette",
    }
    json.dumps(result["metadata"])


def test_split_building_instances_can_supplement_projection_coverage():
    projection = np.zeros((80, 100), dtype=bool)
    projection[20:70, 20:80] = True
    left = np.zeros_like(projection)
    left[22:68, 22:50] = True
    right = np.zeros_like(projection)
    right[22:68, 48:78] = True
    unrelated = np.zeros_like(projection)
    unrelated[4:15, 4:15] = True

    result = build_prefit_semantic_guidance(
        {"building_facade": np.stack([left, right, unrelated])},
        projection,
        config=_config(target_association_distance_px=5.0),
    )

    assert result["selected_building_mask"][40, 30]
    assert result["selected_building_mask"][40, 70]
    assert not result["selected_building_mask"][8, 8]
    assert result["metadata"]["roles"]["building"]["selected_instances"] == 2


def test_empty_semantic_stacks_fall_back_to_projection_local_evidence():
    projection = np.zeros((40, 50), dtype=bool)
    projection[15:25, 20:30] = True

    result = build_prefit_semantic_guidance(
        {
            "building": np.zeros((0, 40, 50), dtype=bool),
            "tree": None,
        },
        projection,
        config=_config(search_dilation_px=4),
    )

    assert result["metadata"]["fallback_used"] is True
    assert result["metadata"]["reason"] == "no_projection_local_semantic_instances"
    assert not result["target_semantic_mask"].any()
    assert result["valid_evidence_mask"].any()
    assert not result["valid_evidence_mask"][0, 0]
    assert not any(mask.any() for mask in result["boundary_maps"].values())
    json.dumps(result["metadata"])


def test_empty_projection_returns_neutral_evidence_and_zero_guidance():
    projection = np.zeros((24, 32), dtype=bool)
    building = np.ones_like(projection)

    result = build_prefit_semantic_guidance(
        {"building": building},
        projection,
        config=_config(),
    )

    assert result["metadata"]["reason"] == "empty_projection_mask"
    assert result["metadata"]["fallback_used"] is True
    assert result["valid_evidence_mask"].all()
    assert not result["local_search_mask"].any()
    assert not result["selected_building_mask"].any()
    assert not any(mask.any() for mask in result["boundary_maps"].values())


def test_bad_role_shape_is_ignored_and_overlay_is_rgb_without_mutating_input():
    projection = np.zeros((30, 40), dtype=bool)
    projection[8:25, 10:32] = True
    bad_building = np.ones((2, 10, 10), dtype=bool)
    tree = np.zeros_like(projection)
    tree[10:22, 27:36] = True
    tiny_tree = np.zeros_like(projection)
    tiny_tree[1:4, 1:4] = True

    result = build_prefit_semantic_guidance(
        {
            "building": bad_building,
            "tree": np.stack([tree, tiny_tree])[:, None, :, :],
            "not_a_role": np.ones_like(projection),
        },
        projection,
        config=_config(),
    )

    assert result["metadata"]["inputs"]["building"]["status"] == "ignored"
    assert result["metadata"]["inputs"]["building"]["reason"].startswith(
        "shape_mismatch"
    )
    assert result["metadata"]["unknown_roles"] == ["not_a_role"]
    assert result["metadata"]["roles"]["vegetation"]["raw_instances"] == 1
    assert result["metadata"]["roles"]["vegetation"]["input_instances"] == 2
    assert (
        result["metadata"]["roles"]["vegetation"][
            "discarded_small_instances"
        ]
        == 1
    )
    assert result["excluded_evidence_mask"].any()

    image = np.full((30, 40, 3), 180, dtype=np.uint8)
    before = image.copy()
    overlay = create_prefit_semantic_guidance_overlay(image, result)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, image)
    np.testing.assert_array_equal(image, before)

    with pytest.raises(ValueError):
        create_prefit_semantic_guidance_overlay(
            np.zeros((31, 40, 3), dtype=np.uint8),
            result,
        )


def test_invalid_projection_rank_and_invalid_config_are_rejected():
    with pytest.raises(ValueError):
        build_prefit_semantic_guidance({}, np.zeros((4, 5, 1), dtype=bool))
    with pytest.raises(ValueError):
        PrefitSemanticGuidanceConfig(occluder_dilation_px=-1)


def test_raw_projection_debug_outline_does_not_draw_viewport_closure():
    shape = (80, 100)
    projection = np.zeros(shape, dtype=bool)
    projection[15:, 25:] = True
    guidance = {
        "raw_projection_mask": projection,
        "local_search_mask": np.ones(shape, dtype=bool),
        "boundary_maps": {},
    }

    overlay = create_prefit_semantic_guidance_overlay(
        np.zeros((*shape, 3), dtype=np.uint8),
        guidance,
        line_thickness_px=1,
        draw_legend=False,
    )

    violet = (
        (overlay[:, :, 0] > 60)
        & (overlay[:, :, 0] < 160)
        & (overlay[:, :, 1] > 70)
        & (overlay[:, :, 1] < 180)
        & (overlay[:, :, 2] > 200)
    )
    assert int(violet[:, -1].sum()) <= 4
    assert int(violet[-1, :].sum()) <= 4
    assert violet[15, 30:90].any()
    assert violet[20:70, 25].any()


class _FakeSam3Processor:
    def __init__(self, masks_by_prompt):
        self.masks_by_prompt = masks_by_prompt
        self.set_image_calls = 0
        self.prompt_calls = []

    def set_image(self, image):
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        self.set_image_calls += 1
        return {"image_ready": True}

    def set_text_prompt(self, *, state, prompt):
        assert state["image_ready"] is True
        self.prompt_calls.append(prompt)
        return {"masks": self.masks_by_prompt[prompt]}


def test_pipeline_runs_one_rgb_embedding_for_automatic_prompt_library(
    monkeypatch,
):
    shape = (64, 64)
    projection = np.zeros(shape, dtype=bool)
    projection[18:52, 18:48] = True
    building = np.zeros(shape, dtype=bool)
    building[20:50, 20:46] = True
    sky = np.zeros(shape, dtype=bool)
    sky[:20, :] = True
    tree = np.zeros(shape, dtype=bool)
    tree[15:48, 42:52] = True
    processor = _FakeSam3Processor({
        "building": building[None, :, :],
        "sky": sky[None, :, :],
        "tree": tree[None, :, :],
    })
    monkeypatch.setattr(
        pipeline,
        "ENABLE_MODEL_DEPTH_PREFIT_SEMANTIC_GUIDANCE",
        True,
    )
    monkeypatch.setattr(
        pipeline,
        "MODEL_DEPTH_PREFIT_SEMANTIC_PROMPT_LIBRARY",
        {
            "building": ("building",),
            "sky": ("sky",),
            "vegetation": ("tree",),
        },
    )
    source_rgba = Image.new("RGBA", (64, 64), (20, 40, 60, 0))

    guidance = pipeline._run_model_depth_prefit_semantic_guidance(
        processor=processor,
        image_rgb=source_rgba,
        raw_projection_mask=projection,
        stage="unit_test_before_fit",
    )

    assert processor.set_image_calls == 1
    assert processor.prompt_calls == ["building", "sky", "tree"]
    assert source_rgba.mode == "RGBA"
    assert guidance["valid_evidence_mask"].dtype == bool
    assert guidance["valid_evidence_mask"].shape == shape
    assert not guidance["valid_evidence_mask"][30, 46]
    assert guidance["metadata"]["manual_prompt_required"] is False
    assert guidance["metadata"]["segmentation_available"] is True
    assert guidance["metadata"]["stage"] == "unit_test_before_fit"


def test_pipeline_combines_semantic_and_osm_exclusions_on_same_canvas():
    shape = (40, 50)
    projection = np.zeros(shape, dtype=bool)
    projection[8:34, 10:42] = True
    tree = np.zeros(shape, dtype=bool)
    tree[12:30, 34:40] = True
    guidance = build_prefit_semantic_guidance(
        {"tree": tree},
        projection,
        config=_config(search_dilation_px=8, occluder_dilation_px=1),
    )
    osm_exclusion = np.zeros(shape, dtype=bool)
    osm_exclusion[:, :15] = True

    combined = pipeline._combine_model_depth_fit_evidence(
        guidance,
        shape,
        external_exclusion_mask=osm_exclusion,
    )

    expected = guidance["valid_evidence_mask"] & ~osm_exclusion
    np.testing.assert_array_equal(combined["valid_evidence_mask"], expected)
    assert combined["valid_evidence_mask"].shape == shape
    assert combined["valid_evidence_mask"].dtype == bool
    assert combined["metadata"]["external_exclusion_pixels"] == int(
        osm_exclusion.sum()
    )
    assert combined["metadata"]["canvas_preserved"] is True
