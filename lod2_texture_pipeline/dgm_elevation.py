# -*- coding: utf-8 -*-
"""In-memory Thuringia DGM1 sampling and camera-elevation validation."""

from __future__ import annotations

import math
import re
import zipfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from rasterio.io import MemoryFile


class DGMError(RuntimeError):
    """Raised when an official DGM tile cannot be fetched or sampled."""


@dataclass
class _DGMGridTile:
    data: np.ndarray
    left: float
    top: float
    x_resolution: float
    y_resolution: float
    nodata: Optional[float]

    def value_at_center(self, x: float, y: float) -> float:
        col = int(round(
            (float(x) - (self.left + 0.5 * self.x_resolution))
            / self.x_resolution
        ))
        row = int(round(
            ((self.top - 0.5 * self.y_resolution) - float(y))
            / self.y_resolution
        ))
        if (
            row < 0
            or col < 0
            or row >= self.data.shape[0]
            or col >= self.data.shape[1]
        ):
            raise DGMError(
                f"DGM grid center ({x:.2f}, {y:.2f}) lies outside its tile."
            )

        value = float(self.data[row, col])
        if not np.isfinite(value):
            raise DGMError(f"DGM contains a non-finite value at ({x:.2f}, {y:.2f}).")
        if self.nodata is not None and np.isclose(value, self.nodata):
            raise DGMError(f"DGM contains nodata at ({x:.2f}, {y:.2f}).")
        return value


class InMemoryThuringiaDGM1:
    """
    Fetch official DGM1 ZIP bytes and decode their GeoTIFF entirely in memory.

    No tile, ZIP, or extracted raster is persisted on disk. A bounded RAM cache
    avoids fetching the same 1 km tile repeatedly during one building run.
    """

    def __init__(
        self,
        *,
        url_template: str,
        timeout_seconds: float = 30.0,
        max_memory_tiles: int = 4,
        expected_horizontal_epsg: int = 25832,
        expected_vertical_epsg: int = 7837,
        session=None,
    ):
        self.url_template = str(url_template)
        self.timeout_seconds = float(timeout_seconds)
        self.max_memory_tiles = max(1, int(max_memory_tiles))
        self.expected_horizontal_epsg = int(expected_horizontal_epsg)
        self.expected_vertical_epsg = int(expected_vertical_epsg)
        self.session = session if session is not None else requests.Session()
        self._tiles: OrderedDict[tuple[int, int], _DGMGridTile] = OrderedDict()
        self._tile_errors: dict[tuple[int, int], str] = {}

    @staticmethod
    def _tile_key_for_grid_center(x: float, y: float) -> tuple[int, int]:
        return int(math.floor(float(x) / 1000.0)), int(
            math.floor(float(y) / 1000.0)
        )

    def _tile_url(self, key: tuple[int, int]) -> str:
        easting_km, northing_km = key
        return self.url_template.format(
            easting_km=easting_km,
            northing_km=northing_km,
        )

    def _fetch_tile(self, key: tuple[int, int]) -> _DGMGridTile:
        url = self._tile_url(key)
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            raise DGMError(f"official DGM tile request failed for {url}: {exc}") from exc

        payload = bytes(response.content)
        if not payload:
            raise DGMError(f"official DGM tile response was empty for {url}")

        try:
            with zipfile.ZipFile(BytesIO(payload), "r") as archive:
                names = archive.namelist()
                tif_name = next(
                    (name for name in names if name.lower().endswith((".tif", ".tiff"))),
                    None,
                )
                if tif_name is None:
                    raise DGMError(f"DGM ZIP contains no GeoTIFF: {url}")
                tif_bytes = archive.read(tif_name)

                meta_name = next(
                    (name for name in names if name.lower().endswith(".meta")),
                    None,
                )
                if meta_name is not None:
                    meta_text = archive.read(meta_name).decode(
                        "utf-8", errors="replace"
                    )
                    vertical_match = re.search(
                        r"EPSG-Code\s+Hoehe\s*:\s*(\d+)",
                        meta_text,
                        flags=re.IGNORECASE,
                    )
                    if (
                        vertical_match is not None
                        and int(vertical_match.group(1))
                        != self.expected_vertical_epsg
                    ):
                        raise DGMError(
                            "unexpected DGM vertical CRS "
                            f"EPSG:{vertical_match.group(1)} in {url}"
                        )
        except DGMError:
            raise
        except Exception as exc:
            raise DGMError(f"could not decode official DGM ZIP {url}: {exc}") from exc

        try:
            with MemoryFile(tif_bytes) as memory_file:
                with memory_file.open() as dataset:
                    raster_epsg = (
                        dataset.crs.to_epsg() if dataset.crs is not None else None
                    )
                    if raster_epsg != self.expected_horizontal_epsg:
                        raise DGMError(
                            "unexpected DGM horizontal CRS "
                            f"{dataset.crs!s}; expected EPSG:"
                            f"{self.expected_horizontal_epsg}"
                        )
                    transform = dataset.transform
                    if (
                        abs(float(transform.b)) > 1.0e-9
                        or abs(float(transform.d)) > 1.0e-9
                        or float(transform.a) <= 0.0
                        or float(transform.e) >= 0.0
                    ):
                        raise DGMError("DGM GeoTIFF has an unsupported grid transform.")
                    data = dataset.read(1).astype(np.float32, copy=False)
                    tile = _DGMGridTile(
                        data=data,
                        left=float(transform.c),
                        top=float(transform.f),
                        x_resolution=float(transform.a),
                        y_resolution=float(-transform.e),
                        nodata=(
                            None
                            if dataset.nodata is None
                            else float(dataset.nodata)
                        ),
                    )
        except DGMError:
            raise
        except Exception as exc:
            raise DGMError(f"could not read in-memory DGM GeoTIFF {url}: {exc}") from exc

        return tile

    def _get_tile(self, key: tuple[int, int]) -> _DGMGridTile:
        previous_error = self._tile_errors.get(key)
        if previous_error is not None:
            raise DGMError(previous_error)
        tile = self._tiles.pop(key, None)
        if tile is None:
            try:
                tile = self._fetch_tile(key)
            except DGMError as exc:
                self._tile_errors[key] = str(exc)
                raise
        self._tiles[key] = tile
        while len(self._tiles) > self.max_memory_tiles:
            self._tiles.popitem(last=False)
        return tile

    def _grid_center_value(self, x: float, y: float) -> float:
        key = self._tile_key_for_grid_center(x, y)
        return self._get_tile(key).value_at_center(x, y)

    def sample(self, x: float, y: float) -> float:
        """Bilinearly sample DGM1 at an EPSG:25832 coordinate."""
        x = float(x)
        y = float(y)
        if not np.isfinite([x, y]).all():
            raise DGMError("DGM sample coordinate is not finite.")

        x0 = math.floor(x - 0.5) + 0.5
        y0 = math.floor(y - 0.5) + 0.5
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        tx = float(np.clip(x - x0, 0.0, 1.0))
        ty = float(np.clip(y - y0, 0.0, 1.0))

        weighted_centers = (
            (x0, y0, (1.0 - tx) * (1.0 - ty)),
            (x1, y0, tx * (1.0 - ty)),
            (x0, y1, (1.0 - tx) * ty),
            (x1, y1, tx * ty),
        )
        value = 0.0
        for grid_x, grid_y, weight in weighted_centers:
            if weight <= 1.0e-12:
                continue
            value += weight * self._grid_center_value(grid_x, grid_y)
        if not np.isfinite(value):
            raise DGMError(f"DGM interpolation failed at ({x:.3f}, {y:.3f}).")
        return float(value)


@dataclass
class DGMBaseValidation:
    consistent: bool
    reason: str
    vertex_count: int
    sampled_count: int
    inlier_count: int
    inlier_fraction: float
    sampled_indices: list[int]
    inlier_indices: list[int]
    outlier_indices: list[int]
    missing_indices: list[int]
    model_elevations_m: list[float]
    dgm_elevations_m: list[Optional[float]]
    differences_m: list[Optional[float]]
    sample_errors: list[Optional[str]]
    inlier_mean_difference_m: Optional[float]
    inlier_median_difference_m: Optional[float]
    inlier_median_absolute_difference_m: Optional[float]
    inlier_max_absolute_difference_m: Optional[float]
    robust_outlier_limit_m: Optional[float]

    def as_dict(self) -> dict:
        return asdict(self)


def unique_base_vertices_from_edges(
    corners,
    base_edges,
    id_to_idx,
) -> np.ndarray:
    """Return each model base node once as an ``N x 3`` array."""
    unique_ids = []
    seen = set()
    for source, target in base_edges or []:
        for node_id in (source, target):
            if node_id in seen or node_id not in id_to_idx:
                continue
            seen.add(node_id)
            unique_ids.append(node_id)
    if not unique_ids:
        return np.empty((0, 3), dtype=np.float64)
    values = np.asarray(
        [corners[int(id_to_idx[node_id])] for node_id in unique_ids],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return values[:, :3]


def validate_dgm_base_vertices(
    sampler,
    base_vertices,
    *,
    minimum_inlier_vertices: int = 3,
    minimum_inlier_fraction: float = 0.66,
    outlier_mad_scale: float = 3.5,
    outlier_minimum_deviation_m: float = 0.50,
    maximum_inlier_absolute_difference_m: float = 0.75,
    maximum_median_absolute_difference_m: float = 0.50,
) -> DGMBaseValidation:
    """
    Compare model base vertices with DGM at the same XY coordinates.

    A robust median/MAD test removes isolated geometric outliers. Remaining
    vertices must also agree in absolute elevation, preventing a coherent but
    incompatible vertical offset from being accepted.
    """
    vertices = np.asarray(base_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        vertices = np.empty((0, 3), dtype=np.float64)
    finite_rows = np.isfinite(vertices[:, :3]).all(axis=1)
    vertices = vertices[finite_rows]
    vertex_count = int(len(vertices))
    model_values = [float(value) for value in vertices[:, 2]]
    dgm_values: list[Optional[float]] = [None] * vertex_count
    differences: list[Optional[float]] = [None] * vertex_count
    sample_errors: list[Optional[str]] = [None] * vertex_count
    sampled_indices = []
    missing_indices = []

    for index, vertex in enumerate(vertices):
        try:
            dgm_z = float(sampler.sample(vertex[0], vertex[1]))
        except Exception as exc:
            sample_errors[index] = f"{type(exc).__name__}: {exc}"
            missing_indices.append(index)
            continue
        dgm_values[index] = dgm_z
        differences[index] = dgm_z - float(vertex[2])
        sampled_indices.append(index)

    sampled_differences = np.asarray(
        [differences[index] for index in sampled_indices],
        dtype=np.float64,
    )
    robust_limit = None
    inlier_indices: list[int] = []
    if sampled_differences.size:
        center = float(np.median(sampled_differences))
        mad = float(np.median(np.abs(sampled_differences - center)))
        robust_limit = max(
            float(outlier_minimum_deviation_m),
            float(outlier_mad_scale) * 1.4826 * mad,
        )
        robust_inlier = (
            np.abs(sampled_differences - center) <= robust_limit + 1.0e-9
        )
        absolute_inlier = (
            np.abs(sampled_differences)
            <= float(maximum_inlier_absolute_difference_m) + 1.0e-9
        )
        inlier_indices = [
            int(sampled_indices[position])
            for position in np.flatnonzero(robust_inlier & absolute_inlier)
        ]

    inlier_set = set(inlier_indices)
    outlier_indices = [
        int(index) for index in sampled_indices if index not in inlier_set
    ]
    inlier_differences = np.asarray(
        [differences[index] for index in inlier_indices],
        dtype=np.float64,
    )
    inlier_count = int(len(inlier_indices))
    inlier_fraction = (
        float(inlier_count) / float(vertex_count) if vertex_count else 0.0
    )
    mean_difference = (
        float(np.mean(inlier_differences)) if inlier_count else None
    )
    median_difference = (
        float(np.median(inlier_differences)) if inlier_count else None
    )
    median_absolute_difference = (
        float(np.median(np.abs(inlier_differences))) if inlier_count else None
    )
    max_absolute_difference = (
        float(np.max(np.abs(inlier_differences))) if inlier_count else None
    )

    minimum_inlier_vertices = max(1, int(minimum_inlier_vertices))
    minimum_inlier_fraction = float(minimum_inlier_fraction)
    if vertex_count < minimum_inlier_vertices:
        consistent = False
        reason = "too_few_model_base_vertices"
    elif len(sampled_indices) < minimum_inlier_vertices:
        consistent = False
        reason = "too_few_dgm_base_samples"
    elif inlier_count < minimum_inlier_vertices:
        consistent = False
        reason = "too_few_consistent_base_vertices"
    elif inlier_fraction + 1.0e-9 < minimum_inlier_fraction:
        consistent = False
        reason = "consistent_base_vertex_fraction_too_low"
    elif (
        median_absolute_difference is None
        or median_absolute_difference
        > float(maximum_median_absolute_difference_m) + 1.0e-9
    ):
        consistent = False
        reason = "median_base_elevation_difference_too_large"
    else:
        consistent = True
        reason = "base_vertices_consistent_after_outlier_rejection"

    return DGMBaseValidation(
        consistent=bool(consistent),
        reason=reason,
        vertex_count=vertex_count,
        sampled_count=int(len(sampled_indices)),
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        sampled_indices=[int(index) for index in sampled_indices],
        inlier_indices=inlier_indices,
        outlier_indices=outlier_indices,
        missing_indices=[int(index) for index in missing_indices],
        model_elevations_m=model_values,
        dgm_elevations_m=dgm_values,
        differences_m=differences,
        sample_errors=sample_errors,
        inlier_mean_difference_m=mean_difference,
        inlier_median_difference_m=median_difference,
        inlier_median_absolute_difference_m=median_absolute_difference,
        inlier_max_absolute_difference_m=max_absolute_difference,
        robust_outlier_limit_m=robust_limit,
    )


@dataclass
class CameraElevationDecision:
    used_dgm: bool
    reason: str
    camera_z_m: float
    ground_z_m: Optional[float]
    fallback_camera_z_m: float
    difference_from_fallback_m: Optional[float]
    camera_height_m: float
    x: float
    y: float

    def as_dict(self) -> dict:
        return asdict(self)


class CameraElevationResolver:
    """Resolve camera Z from validated DGM1 or the legacy building-base level."""

    def __init__(
        self,
        *,
        building_label: str,
        sampler,
        base_vertices,
        fallback_base_z: float,
        camera_height_m: float,
        enabled: bool = True,
        minimum_inlier_vertices: int = 3,
        minimum_inlier_fraction: float = 0.66,
        outlier_mad_scale: float = 3.5,
        outlier_minimum_deviation_m: float = 0.50,
        maximum_inlier_absolute_difference_m: float = 0.75,
        maximum_median_absolute_difference_m: float = 0.50,
        emit_diagnostics: bool = True,
    ):
        self.building_label = str(building_label)
        self.sampler = sampler
        self.fallback_base_z = float(fallback_base_z)
        self.camera_height_m = float(camera_height_m)
        self.enabled = bool(enabled)
        self.emit_diagnostics = bool(emit_diagnostics)
        self._decision_cache: dict[tuple[float, float], CameraElevationDecision] = {}

        if not self.enabled:
            self.validation = DGMBaseValidation(
                consistent=False,
                reason="disabled",
                vertex_count=int(len(np.asarray(base_vertices))),
                sampled_count=0,
                inlier_count=0,
                inlier_fraction=0.0,
                sampled_indices=[],
                inlier_indices=[],
                outlier_indices=[],
                missing_indices=[],
                model_elevations_m=[],
                dgm_elevations_m=[],
                differences_m=[],
                sample_errors=[],
                inlier_mean_difference_m=None,
                inlier_median_difference_m=None,
                inlier_median_absolute_difference_m=None,
                inlier_max_absolute_difference_m=None,
                robust_outlier_limit_m=None,
            )
        elif self.sampler is None:
            self.validation = DGMBaseValidation(
                consistent=False,
                reason="sampler_unavailable",
                vertex_count=int(len(np.asarray(base_vertices))),
                sampled_count=0,
                inlier_count=0,
                inlier_fraction=0.0,
                sampled_indices=[],
                inlier_indices=[],
                outlier_indices=[],
                missing_indices=[],
                model_elevations_m=[],
                dgm_elevations_m=[],
                differences_m=[],
                sample_errors=[],
                inlier_mean_difference_m=None,
                inlier_median_difference_m=None,
                inlier_median_absolute_difference_m=None,
                inlier_max_absolute_difference_m=None,
                robust_outlier_limit_m=None,
            )
        else:
            self.validation = validate_dgm_base_vertices(
                self.sampler,
                base_vertices,
                minimum_inlier_vertices=minimum_inlier_vertices,
                minimum_inlier_fraction=minimum_inlier_fraction,
                outlier_mad_scale=outlier_mad_scale,
                outlier_minimum_deviation_m=outlier_minimum_deviation_m,
                maximum_inlier_absolute_difference_m=(
                    maximum_inlier_absolute_difference_m
                ),
                maximum_median_absolute_difference_m=(
                    maximum_median_absolute_difference_m
                ),
            )

        if self.emit_diagnostics:
            self._print_validation()

    def _print_validation(self) -> None:
        validation = self.validation
        differences = []
        inlier_set = set(validation.inlier_indices)
        for index, difference in enumerate(validation.differences_m):
            if difference is None:
                differences.append(f"v{index}=missing")
                continue
            suffix = "" if index in inlier_set else " (excluded)"
            differences.append(f"v{index}={difference:+.3f}m{suffix}")

        status = "DGM ACCEPTED" if validation.consistent else "DGM NOT USED"
        print(
            f"[DGM] {self.building_label} | {status} | "
            f"base samples={validation.sampled_count}/{validation.vertex_count}, "
            f"inliers={validation.inlier_count} "
            f"({100.0 * validation.inlier_fraction:.1f}%) | "
            f"reason={validation.reason}"
        )
        if differences:
            print(f"      DGM-model base differences: {', '.join(differences)}")
        errors = [
            error for error in validation.sample_errors if error is not None
        ]
        if errors:
            print(f"      DGM sample failure: {errors[0]}")
        if validation.inlier_count:
            print(
                "      retained differences: "
                f"mean={validation.inlier_mean_difference_m:+.3f}m, "
                f"median={validation.inlier_median_difference_m:+.3f}m, "
                f"max_abs={validation.inlier_max_absolute_difference_m:.3f}m"
            )

    def resolve(
        self,
        x: float,
        y: float,
        *,
        source_label: str = "",
    ) -> CameraElevationDecision:
        x = float(x)
        y = float(y)
        key = (round(x, 3), round(y, 3))
        cached = self._decision_cache.get(key)
        if cached is not None:
            return cached

        fallback_z = self.fallback_base_z + self.camera_height_m
        if not self.validation.consistent:
            decision = CameraElevationDecision(
                used_dgm=False,
                reason=f"base_validation_failed:{self.validation.reason}",
                camera_z_m=fallback_z,
                ground_z_m=None,
                fallback_camera_z_m=fallback_z,
                difference_from_fallback_m=None,
                camera_height_m=self.camera_height_m,
                x=x,
                y=y,
            )
        else:
            try:
                ground_z = float(self.sampler.sample(x, y))
                camera_z = ground_z + self.camera_height_m
                decision = CameraElevationDecision(
                    used_dgm=True,
                    reason="validated_dgm_camera_ground",
                    camera_z_m=camera_z,
                    ground_z_m=ground_z,
                    fallback_camera_z_m=fallback_z,
                    difference_from_fallback_m=camera_z - fallback_z,
                    camera_height_m=self.camera_height_m,
                    x=x,
                    y=y,
                )
            except Exception as exc:
                decision = CameraElevationDecision(
                    used_dgm=False,
                    reason=f"camera_dgm_sample_failed:{type(exc).__name__}:{exc}",
                    camera_z_m=fallback_z,
                    ground_z_m=None,
                    fallback_camera_z_m=fallback_z,
                    difference_from_fallback_m=None,
                    camera_height_m=self.camera_height_m,
                    x=x,
                    y=y,
                )

        self._decision_cache[key] = decision
        if self.emit_diagnostics:
            label = f" {source_label}" if source_label else ""
            if decision.used_dgm:
                print(
                    f"[DGM] camera{label} | USED | ground="
                    f"{decision.ground_z_m:.3f}m, camera="
                    f"{decision.camera_z_m:.3f}m | legacy="
                    f"{decision.fallback_camera_z_m:.3f}m | difference="
                    f"{decision.difference_from_fallback_m:+.3f}m"
                )
            else:
                print(
                    f"[DGM] camera{label} | NOT USED | legacy="
                    f"{decision.fallback_camera_z_m:.3f}m | "
                    "difference=unavailable | "
                    f"reason={decision.reason}"
                )
        return decision

    def metadata(self) -> dict:
        return {
            "enabled": self.enabled,
            "building_label": self.building_label,
            "camera_height_m": self.camera_height_m,
            "fallback_base_z_m": self.fallback_base_z,
            "base_validation": self.validation.as_dict(),
        }
