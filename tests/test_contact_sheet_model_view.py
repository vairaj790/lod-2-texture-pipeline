import numpy as np

from lod2_texture_pipeline.pipeline import _debug_model_view_basis


def _view_row(camera_xyz, wall_x):
    return {
        "camera_utm_xyz": camera_xyz,
        "wall_quad_xyz_b1b2t2t1": [
            [wall_x, 0.0, 0.0],
            [wall_x, 4.0, 0.0],
            [wall_x, 4.0, 3.0],
            [wall_x, 0.0, 3.0],
        ],
    }


def test_contact_sheet_model_basis_does_not_follow_streetview_camera():
    from_west = [_view_row([-100.0, 0.0, 2.5], 0.0)]
    from_east = [_view_row([100.0, 0.0, 2.5], 20.0)]

    west_basis = _debug_model_view_basis(from_west)
    east_basis = _debug_model_view_basis(from_east)
    empty_basis = _debug_model_view_basis([])

    for west, east, empty in zip(west_basis, east_basis, empty_basis):
        assert np.allclose(west, east)
        assert np.allclose(west, empty)


def test_contact_sheet_model_basis_is_orthonormal_and_looks_downward():
    camera_dir, right, up = _debug_model_view_basis([])

    assert np.isclose(np.linalg.norm(camera_dir), 1.0)
    assert np.isclose(np.linalg.norm(right), 1.0)
    assert np.isclose(np.linalg.norm(up), 1.0)
    assert np.isclose(np.dot(camera_dir, right), 0.0)
    assert np.isclose(np.dot(camera_dir, up), 0.0)
    assert np.isclose(np.dot(right, up), 0.0)
    assert camera_dir[2] < 0.0
