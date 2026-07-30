import numpy as np

import lod2_texture_pipeline.inpainting as inpainting


class _FakeLamaInpainter:
    def __init__(self):
        self.mask = None

    def infer(self, image_bgr, mask_u8):
        self.mask = np.asarray(mask_u8).copy()
        result = np.asarray(image_bgr).copy()
        result[mask_u8 > 0] = (10, 20, 30)
        return result


def test_lama_hole_mask_uses_explicit_valid_sam_content(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(inpainting, "LAMA_MIN_HOLE_AREA_PX", 1)
    monkeypatch.setattr(inpainting, "LAMA_MASK_DILATE_PX", 0)
    fake_lama = _FakeLamaInpainter()
    monkeypatch.setattr(inpainting, "get_lama_inpainter", lambda: fake_lama)

    height = 80
    width = 100
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = (120, 140, 160)
    rgba[:, :, 3] = 255
    wall_poly = np.array(
        [[10.0, 70.0], [90.0, 70.0], [90.0, 10.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    wall_mask = inpainting.build_wall_region_mask(height, width, wall_poly) > 0
    valid_sam = wall_mask.copy()
    valid_sam[30:51, 40:66] = False
    debug_path = tmp_path / "lama_mask.png"

    filled, hole_mask = inpainting.lama_fill_rectified_wall(
        ortho_rgba=rgba,
        wall_poly_px=wall_poly,
        debug_mask_path=str(debug_path),
        valid_content_mask=valid_sam,
    )

    expected_holes = wall_mask & ~valid_sam
    np.testing.assert_array_equal(hole_mask > 0, expected_holes)
    np.testing.assert_array_equal(fake_lama.mask > 0, expected_holes)
    assert np.all(filled[expected_holes, :3] == np.array([30, 20, 10]))
    assert np.all(filled[expected_holes, 3] == 255)
    assert debug_path.is_file()


def test_lama_rejects_mismatched_valid_content_mask():
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    wall_poly = np.array(
        [[1.0, 18.0], [28.0, 18.0], [28.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )

    try:
        inpainting.lama_fill_rectified_wall(
            ortho_rgba=rgba,
            wall_poly_px=wall_poly,
            valid_content_mask=np.zeros((10, 10), dtype=bool),
        )
    except ValueError as exc:
        assert "valid_content_mask" in str(exc)
    else:
        raise AssertionError("Expected mismatched valid_content_mask to fail")
