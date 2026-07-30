import numpy as np

from lod2_texture_pipeline.facade_refinement import (
    constrain_post_rectification_sam_mask,
)


def test_post_rectification_sam_is_clipped_to_projection():
    allowed = np.zeros((8, 8), dtype=bool)
    allowed[2:6, 2:6] = True
    sam = np.zeros_like(allowed)
    sam[1:6, 1:6] = True

    result = constrain_post_rectification_sam_mask(
        sam,
        allowed,
        minimum_pixels=4,
        minimum_wall_coverage=0.25,
    )

    assert result["accepted"]
    assert result["outside_pixels_removed"] == 9
    assert not np.any(result["clipped_mask"] & ~allowed)
    np.testing.assert_array_equal(result["effective_refinement_mask"], allowed)


def test_empty_sam_falls_back_to_nonempty_projection_mask():
    allowed = np.zeros((10, 10), dtype=bool)
    allowed[2:8, 3:7] = True

    result = constrain_post_rectification_sam_mask(
        np.zeros_like(allowed),
        allowed,
        minimum_pixels=5,
        minimum_wall_coverage=0.2,
    )

    assert not result["accepted"]
    assert result["fallback_to_projection_mask"]
    assert result["reason"] == "empty_sam_mask_inside_projection"
    np.testing.assert_array_equal(result["effective_refinement_mask"], allowed)


def test_small_sam_mask_is_not_allowed_to_replace_projection():
    allowed = np.ones((10, 10), dtype=bool)
    sam = np.zeros_like(allowed)
    sam[4:6, 4:6] = True

    result = constrain_post_rectification_sam_mask(
        sam,
        allowed,
        minimum_pixels=1,
        minimum_wall_coverage=0.2,
    )

    assert not result["accepted"]
    assert result["reason"] == "insufficient_wall_coverage"
    np.testing.assert_array_equal(result["effective_refinement_mask"], allowed)
