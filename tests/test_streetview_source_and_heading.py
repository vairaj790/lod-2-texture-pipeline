import numpy as np
from PIL import Image

import lod2_texture_pipeline.projection as projection
import lod2_texture_pipeline.streetview as streetview


class _MetadataResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_metadata_search_uses_outdoor_and_rejects_non_google(monkeypatch):
    calls = []
    payloads = [
        {
            "status": "OK",
            "pano_id": "ugc",
            "copyright": "(c) Individual Contributor",
            "location": {"lat": 50.0, "lng": 11.0},
        },
        {
            "status": "OK",
            "pano_id": "google",
            "copyright": "(c) 2026 Google",
            "date": "2026-01",
            "location": {"lat": 50.0, "lng": 11.0},
        },
    ]

    def fake_get(url, params, timeout):
        calls.append((url, dict(params), timeout))
        return _MetadataResponse(payloads.pop(0))

    monkeypatch.setattr(streetview, "ENABLE_STREETVIEW_CACHE", False)
    monkeypatch.setattr(streetview.requests, "get", fake_get)

    rejected = streetview.get_nearest_pano(50.0, 11.0, "key")
    accepted = streetview.get_nearest_pano(50.0, 11.001, "key")

    assert rejected is None
    assert accepted["pano_id"] == "google"
    assert all(call[1]["source"] == "outdoor" for call in calls)


def test_panorama_record_preserves_provider_metadata():
    class IdentityTransform:
        @staticmethod
        def transform(x, y):
            return x, y

    record = streetview._pano_record_from_metadata(
        {
            "status": "OK",
            "pano_id": "google",
            "copyright": "(c) 2024 Google",
            "date": "2024-08",
            "location": {"lat": 50.5, "lng": 11.5},
        },
        IdentityTransform(),
    )

    assert record["copyright"] == "(c) 2024 Google"
    assert record["date"] == "2024-08"
    assert record["imagery_provider"] == "google"
    assert record["search_source"] == "outdoor"


def test_grid_heading_is_converted_to_true_north_at_jena():
    camera_xy = np.array([681130.6634899076, 5645464.781410679])
    true_heading = streetview.grid_heading_to_true_deg(camera_xy, 0.0)

    assert 1.9 < true_heading < 2.1


def test_group_fetch_uses_true_heading_but_projection_keeps_grid_heading(monkeypatch):
    y_wall = 5645494.781410679
    x_center = 681130.6634899076
    base_z = 175.0
    outline_xyz = np.array([
        [x_center - 5.0, y_wall, base_z],
        [x_center + 5.0, y_wall, base_z],
        [x_center + 5.0, y_wall, base_z + 10.0],
        [x_center - 5.0, y_wall, base_z + 10.0],
    ])
    outline_m = np.array([
        [-5.0, 0.0],
        [5.0, 0.0],
        [5.0, 10.0],
        [-5.0, 10.0],
    ])

    def to_xyz(uv):
        values = np.asarray(uv, dtype=np.float64)
        if values.ndim == 1:
            return np.array([
                x_center + values[0],
                y_wall,
                base_z + values[1],
            ])
        return np.column_stack([
            x_center + values[:, 0],
            np.full(len(values), y_wall),
            base_z + values[:, 1],
        ])

    geom = {
        "frame": {
            "origin": np.array([x_center, y_wall, base_z]),
            "u_dir": np.array([1.0, 0.0, 0.0]),
            "normal_xy": np.array([0.0, -1.0]),
            "to_xyz": to_xyz,
        },
        "outline_xyz": outline_xyz,
        "outline_m": outline_m,
    }
    requested_headings = []

    def fake_fetch(pano_id, heading, pitch, fov, api_key, size):
        requested_headings.append(float(heading))
        return Image.new("RGB", (640, 640), (120, 120, 120)), "cache://google", b"", "image/jpeg"

    def fake_wireframe_fit(source, _outline_xyz, _facade_tag, _source_index):
        source["wireframe_fit_H"] = np.eye(3)
        source["wireframe_fit"] = {"applied": False, "reason": "test_identity"}
        source["wireframe_fit_overlay"] = None
        source["effective_camera_fit"] = {"attempted": False, "accepted": False}

    monkeypatch.setattr(projection, "fetch_sv_image_by_id", fake_fetch)
    monkeypatch.setattr(projection, "_fit_raw_source_wireframe", fake_wireframe_fit)

    result = projection.select_facade_source_from_panos(
        geom=geom,
        pano_candidates=[{
            "rec": {
                "pano_id": "google",
                "utm": (x_center, y_wall - 30.0),
            },
            "u_clamped": 0.0,
        }],
        base_z=base_z,
        rect_xyz=outline_xyz,
        outline_xyz=outline_xyz,
        facade_tag="heading_test",
        img_size="640x640",
    )

    assert result is not None
    source = result["sources"][0]
    assert abs(source["projection_heading"]) < 1.0e-8
    assert 1.9 < source["heading"] < 2.1
    assert np.isclose(requested_headings[0], source["heading"])
    assert np.allclose(source["Rwc"][2, :2], [0.0, np.cos(np.radians(source["pitch"]))])
