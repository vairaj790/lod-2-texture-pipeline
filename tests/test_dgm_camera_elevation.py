from io import BytesIO
import zipfile

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from lod2_texture_pipeline.dgm_elevation import (
    CameraElevationResolver,
    InMemoryThuringiaDGM1,
    unique_base_vertices_from_edges,
    validate_dgm_base_vertices,
)


class _CoordinateSampler:
    def __init__(self, values):
        self.values = {
            (round(float(x), 3), round(float(y), 3)): float(z)
            for (x, y), z in values.items()
        }
        self.calls = []

    def sample(self, x, y):
        key = (round(float(x), 3), round(float(y), 3))
        self.calls.append(key)
        return self.values[key]


def _jena_base_vertices():
    return np.array(
        [
            [681071.625, 5645763.434375, 210.80],
            [681057.7171875, 5645767.04375, 210.73],
            [681054.80625, 5645754.05, 209.61],
            [681068.821875, 5645751.059375, 207.90],
        ],
        dtype=np.float64,
    )


def test_base_validation_accepts_majority_and_excludes_extreme_vertex():
    vertices = _jena_base_vertices()
    sampler = _CoordinateSampler(
        {
            tuple(vertices[0, :2]): 211.171,
            tuple(vertices[1, :2]): 210.857,
            tuple(vertices[2, :2]): 209.882,
            tuple(vertices[3, :2]): 210.071,
        }
    )

    result = validate_dgm_base_vertices(sampler, vertices)

    assert result.consistent is True
    assert result.inlier_indices == [0, 1, 2]
    assert result.outlier_indices == [3]
    assert np.isclose(result.inlier_mean_difference_m, 0.2566666667)
    assert np.isclose(result.inlier_max_absolute_difference_m, 0.371)


def test_camera_uses_raw_dgm_ground_after_base_validation():
    vertices = _jena_base_vertices()
    camera_xy = (681040.17488, 5645750.79601)
    sampler = _CoordinateSampler(
        {
            tuple(vertices[0, :2]): 211.171,
            tuple(vertices[1, :2]): 210.857,
            tuple(vertices[2, :2]): 209.882,
            tuple(vertices[3, :2]): 210.071,
            camera_xy: 205.740,
        }
    )
    resolver = CameraElevationResolver(
        building_label="test",
        sampler=sampler,
        base_vertices=vertices,
        fallback_base_z=209.76,
        camera_height_m=2.5,
        emit_diagnostics=False,
    )

    decision = resolver.resolve(*camera_xy)

    assert decision.used_dgm is True
    assert np.isclose(decision.ground_z_m, 205.74)
    assert np.isclose(decision.camera_z_m, 208.24)
    assert np.isclose(decision.fallback_camera_z_m, 212.26)
    assert np.isclose(decision.difference_from_fallback_m, -4.02)


def test_incompatible_base_heights_keep_legacy_camera_elevation():
    vertices = np.array(
        [
            [0.0, 0.0, 100.0],
            [10.0, 0.0, 100.0],
            [10.0, 10.0, 100.0],
            [0.0, 10.0, 100.0],
        ],
        dtype=np.float64,
    )
    sampler = _CoordinateSampler(
        {
            (0.0, 0.0): 102.0,
            (10.0, 0.0): 102.0,
            (10.0, 10.0): 102.0,
            (0.0, 10.0): 102.0,
        }
    )
    resolver = CameraElevationResolver(
        building_label="test",
        sampler=sampler,
        base_vertices=vertices,
        fallback_base_z=100.0,
        camera_height_m=2.5,
        emit_diagnostics=False,
    )

    decision = resolver.resolve(5.0, -5.0)

    assert resolver.validation.consistent is False
    assert decision.used_dgm is False
    assert decision.camera_z_m == 102.5
    assert decision.ground_z_m is None
    assert (5.0, -5.0) not in sampler.calls


def test_unique_base_vertices_are_not_weighted_by_duplicate_edge_endpoints():
    corners = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.1],
            [1.0, 1.0, 10.2],
            [0.0, 1.0, 10.3],
        ],
        dtype=np.float64,
    )
    edges = [(10, 11), (11, 12), (12, 13), (13, 10)]
    id_to_idx = {10: 0, 11: 1, 12: 2, 13: 3}

    result = unique_base_vertices_from_edges(corners, edges, id_to_idx)

    assert result.shape == (4, 3)
    assert np.allclose(result, corners)


class _MemoryResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


class _MemorySession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return _MemoryResponse(self.payload)


def _dgm_zip_bytes():
    data = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:25832",
            transform=from_origin(1000.0, 2000.0, 1.0, 1.0),
            nodata=-9999.0,
        ) as dataset:
            dataset.write(data, 1)
        tif_bytes = memory_file.read()

    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dgm_test.tif", tif_bytes)
        archive.writestr(
            "dgm_test.meta",
            "EPSG-Code Lage: 25832\nEPSG-Code Hoehe: 7837\n",
        )
    return output.getvalue()


def test_official_tile_zip_is_sampled_and_cached_only_in_memory():
    session = _MemorySession(_dgm_zip_bytes())
    sampler = InMemoryThuringiaDGM1(
        url_template="https://example.test/{easting_km}/{northing_km}.zip",
        session=session,
        max_memory_tiles=1,
    )

    first = sampler.sample(1001.0, 1999.0)
    second = sampler.sample(1001.0, 1999.0)

    assert np.isclose(first, 25.0)
    assert np.isclose(second, 25.0)
    assert session.calls == [("https://example.test/1/1.zip", 30.0)]
