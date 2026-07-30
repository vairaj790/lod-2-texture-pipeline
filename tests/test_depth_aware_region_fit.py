import cv2
import numpy as np

from lod2_texture_pipeline.depth_aware_region_fit import (
    DepthAwareRegionFitConfig,
    depth_discontinuity_edges,
    fit_depth_aware_segmentation_region,
    visible_group_mask_from_depth,
)


def _rectangle_fixture():
    height, width = 320, 420
    outline = np.array(
        [[110.0, 85.0], [285.0, 85.0], [285.0, 250.0], [110.0, 250.0]],
        dtype=np.float64,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [outline.astype(np.int32)], 1)
    return mask > 0, outline


def _test_config():
    return DepthAwareRegionFitConfig(
        max_translation_px=70.0,
        minimum_score_improvement=0.02,
        minimum_iou_improvement=0.02,
        minimum_boundary_improvement=0.02,
        optimizer_seed_count=1,
        optimizer_max_evaluations=150,
    )


def test_region_fit_recovers_similarity_from_filled_segmentation():
    source, outline = _rectangle_fixture()
    center = tuple(outline.mean(axis=0))
    affine = cv2.getRotationMatrix2D(center, -2.5, 1.07)
    affine[:, 2] += np.array([31.0, -19.0], dtype=np.float64)
    expected_homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    target = cv2.warpPerspective(
        source.astype(np.uint8),
        expected_homography,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_NEAREST,
    ) > 0

    result = fit_depth_aware_segmentation_region(
        target,
        source,
        outline,
        config=_test_config(),
    )

    assert result["applied"] is True
    assert result["metrics_after"]["iou"] > 0.95
    assert result["metrics_after"]["iou"] > result["metrics_before"]["iou"] + 0.30
    assert abs(result["transform"]["scale"] - 1.07) < 0.02
    assert abs(result["transform"]["tx"] - 31.0) < 3.0
    assert abs(result["transform"]["ty"] + 19.0) < 3.0


def test_region_fit_keeps_an_already_aligned_projection():
    source, outline = _rectangle_fixture()

    result = fit_depth_aware_segmentation_region(
        source,
        source,
        outline,
        config=_test_config(),
    )

    assert result["applied"] is False
    assert np.allclose(result["homography"], np.eye(3))
    assert np.allclose(result["fitted_points"], outline)


def test_region_fit_rejects_segmentation_covering_the_canvas():
    source, outline = _rectangle_fixture()
    target = np.ones_like(source, dtype=bool)

    result = fit_depth_aware_segmentation_region(
        target,
        source,
        outline,
        config=_test_config(),
    )

    assert result["applied"] is False
    assert result["reason"] == "segmentation_covers_most_of_canvas"
    assert np.allclose(result["homography"], np.eye(3))


def test_region_fit_ignores_nearby_component_without_model_overlap():
    source, outline = _rectangle_fixture()
    target = source.copy()
    target[25:65, 325:390] = True

    result = fit_depth_aware_segmentation_region(
        target,
        source,
        outline,
        config=_test_config(),
    )

    assert not result["target_mask"][40, 350]
    assert result["target_mask"][150, 180]
    assert result["applied"] is False


def test_depth_visibility_keeps_only_group_pixels_at_the_full_model_surface():
    full = np.full((8, 10), np.nan, dtype=np.float32)
    group = np.full((8, 10), np.nan, dtype=np.float32)
    full[1:7, 1:9] = 10.0
    group[2:6, 2:8] = 10.03
    full[3:5, 4:6] = 7.0

    visible = visible_group_mask_from_depth(full, group, absolute_tolerance_m=0.08)

    assert visible[2, 2]
    assert not visible[3, 4]
    assert int(visible.sum()) == (4 * 6) - (2 * 2)


def test_depth_edges_include_silhouette_and_internal_jump():
    depth = np.full((12, 14), np.nan, dtype=np.float32)
    depth[2:10, 2:12] = 8.0
    depth[2:10, 8:12] = 11.0

    edges = depth_discontinuity_edges(depth, absolute_jump_m=0.30)

    assert edges[5, 2]
    assert edges[5, 7]
    assert edges[5, 8]
    assert not edges[5, 5]
