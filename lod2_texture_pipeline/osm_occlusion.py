# -*- coding: utf-8 -*-
"""OSM-assisted external-building occlusion geometry and depth tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
import trimesh
from pyproj import Transformer
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import polygonize, transform as shapely_transform, triangulate, unary_union

from lod2_texture_pipeline import config as pipeline_config
from lod2_texture_pipeline.config import SOURCE_CRS
from lod2_texture_pipeline.depth_boundary_fit import (
    fit_depth_silhouette_to_image,
    project_semantic_model_boundary_edges,
)
from lod2_texture_pipeline.geojson_io import build_edge_loops_from_gdf, load_3d_geojson
from lod2_texture_pipeline.mesh import build_trimesh_from_surface_face, triangulate_surface
from lod2_texture_pipeline.projection import (
    build_pose_from_heading_pitch,
    render_model_depth_map,
    warp_depth_map_to_canvas,
)
from lod2_texture_pipeline.wireframe_fit import make_production_fit_config


DEFAULT_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACK_ENDPOINTS = (
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OVERPASS_USER_AGENT = "lod2-texture-pipeline-osm-occlusion/1.0"


@dataclass(frozen=True)
class OSMBuilding:
    osm_type: str
    osm_id: int
    footprint: Polygon
    tags: Mapping[str, str]
    height_m: float
    min_height_m: float = 0.0
    part_index: int = 0

    @property
    def key(self) -> str:
        suffix = f"#part-{self.part_index}" if self.part_index else ""
        return f"{self.osm_type}/{self.osm_id}{suffix}"


def _finite_quad(quad) -> bool:
    quad = np.asarray(quad, dtype=np.float64)
    return bool(quad.shape == (4, 3) and np.isfinite(quad).all())


def _wall_quads_by_global_index(gdf, corners, id_to_idx):
    quads = {}
    global_index = 0
    for loop in build_edge_loops_from_gdf(gdf, "wall"):
        ring_edges = list(loop.get("edges", []))
        for index, (source1, target1) in enumerate(ring_edges):
            source2, target2 = ring_edges[(index + 1) % len(ring_edges)]
            node_ids = (source1, target1, source2, target2)
            quad = np.full((4, 3), np.nan, dtype=np.float64)
            if all(node_id in id_to_idx for node_id in node_ids):
                point1a = corners[id_to_idx[source1]]
                point1b = corners[id_to_idx[target1]]
                point2a = corners[id_to_idx[source2]]
                point2b = corners[id_to_idx[target2]]

                def by_z(point0, point1):
                    return (point0, point1) if point0[2] <= point1[2] else (point1, point0)

                bottom1, top1 = by_z(point1a, point1b)
                bottom2, top2 = by_z(point2a, point2b)
                quad = np.vstack([bottom1, bottom2, top2, top1]).astype(np.float64)
            quads[int(global_index)] = quad
            global_index += 1
    return quads


def _model_footprint(edge_groups, corners, id_to_idx, wall_quads):
    lines = []
    for source, target in edge_groups.get("base", []):
        if source not in id_to_idx or target not in id_to_idx:
            continue
        point0 = np.asarray(corners[id_to_idx[source]], dtype=np.float64)
        point1 = np.asarray(corners[id_to_idx[target]], dtype=np.float64)
        if np.isfinite(np.vstack([point0, point1])).all():
            lines.append(LineString([point0[:2], point1[:2]]))
    polygons = list(polygonize(unary_union(lines))) if lines else []
    if polygons:
        footprint = unary_union(polygons)
    else:
        bottom_points = [
            point[:2]
            for quad in wall_quads.values()
            if _finite_quad(quad)
            for point in np.asarray(quad, dtype=np.float64)[:2]
        ]
        if len(bottom_points) < 3:
            raise ValueError("The LoD-2 model has no usable base footprint.")
        footprint = MultiPoint(bottom_points).convex_hull
    if footprint.geom_type == "MultiPolygon":
        footprint = max(footprint.geoms, key=lambda polygon: polygon.area)
    if footprint.is_empty or not footprint.is_valid:
        footprint = footprint.buffer(0)
    if footprint.is_empty:
        raise ValueError("The LoD-2 model footprint could not be reconstructed.")
    return footprint


def _quad_mesh(quad, name):
    quad = np.asarray(quad, dtype=np.float64)
    mesh = trimesh.Trimesh(
        vertices=quad,
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    return str(name), mesh


def _model_boundary_edges_xyz_by_class(edge_groups, corners, id_to_idx):
    corners = np.asarray(corners, dtype=np.float64)
    resolved = {}
    for edge_class in ("roof", "wall", "base"):
        edges = []
        seen = set()
        for source, target in edge_groups.get(edge_class, []):
            if source not in id_to_idx or target not in id_to_idx:
                continue
            edge = np.vstack([
                corners[int(id_to_idx[source])],
                corners[int(id_to_idx[target])],
            ]).astype(np.float64)
            if not np.isfinite(edge).all() or float(np.linalg.norm(edge[1] - edge[0])) < 1.0e-8:
                continue
            key = tuple(sorted(tuple(np.round(point, 7)) for point in edge))
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
        resolved[edge_class] = np.asarray(edges, dtype=np.float64).reshape(-1, 2, 3)
    return resolved


def _full_model_meshes(gdf, corners, edge_groups, id_to_idx, surface_faces, wall_quads):
    meshes = [
        _quad_mesh(quad, f"wall_{wall_index:03d}")
        for wall_index, quad in sorted(wall_quads.items())
        if _finite_quad(quad)
    ]
    explicit_surfaces = [
        surface
        for surface in surface_faces
        if str(surface.get("surface_type", "")).lower() in {"roof", "roof_seam", "base"}
    ]
    has_explicit_roof = any(
        str(surface.get("surface_type", "")).lower() in {"roof", "roof_seam"}
        for surface in explicit_surfaces
    )
    has_explicit_base = any(
        str(surface.get("surface_type", "")).lower() == "base"
        for surface in explicit_surfaces
    )
    for surface_index, surface in enumerate(explicit_surfaces):
        surface_type = str(surface.get("surface_type", "")).lower()
        if (
            surface_type not in {"roof", "roof_seam", "base"}
            or (surface_type in {"roof", "roof_seam"} and not has_explicit_roof)
            or (surface_type == "base" and not has_explicit_base)
        ):
            continue
        mesh, _coordinates = build_trimesh_from_surface_face(corners, surface)
        if mesh is not None:
            meshes.append((f"{surface_type}_{surface_index:03d}", mesh))

    if not has_explicit_roof:
        roof_edges = edge_groups.get("roof", [])
        if roof_edges:
            coordinate_sets, face_sets = triangulate_surface(
                roof_edges,
                corners,
                id_to_idx,
                split_components=True,
            )
            for roof_index, (coordinates, faces) in enumerate(zip(coordinate_sets, face_sets)):
                if coordinates is None or faces is None:
                    continue
                meshes.append((
                    f"roof_{roof_index:03d}",
                    trimesh.Trimesh(vertices=coordinates, faces=faces, process=False),
                ))
    if not has_explicit_base:
        for base_index, loop in enumerate(build_edge_loops_from_gdf(gdf, "base")):
            coordinates, faces = triangulate_surface(loop["edges"], corners, id_to_idx)
            if coordinates is None or faces is None:
                continue
            meshes.append((
                f"base_{base_index:03d}",
                trimesh.Trimesh(vertices=coordinates, faces=faces, process=False),
            ))
    return meshes


def build_model_occlusion_geometry(geojson_path) -> Dict[str, object]:
    """Load the model footprint and target-wall meshes using production indexing."""
    gdf, corners, edge_groups, id_to_idx, _centers, base_z, surface_faces = load_3d_geojson(
        str(geojson_path)
    )
    corners = np.asarray(corners, dtype=np.float64)
    wall_quads = _wall_quads_by_global_index(gdf, corners, id_to_idx)
    footprint = _model_footprint(edge_groups, corners, id_to_idx, wall_quads)
    full_model_meshes = _full_model_meshes(
        gdf,
        corners,
        edge_groups,
        id_to_idx,
        surface_faces,
        wall_quads,
    )
    return {
        "gdf": gdf,
        "corners": corners,
        "edge_groups": edge_groups,
        "id_to_idx": id_to_idx,
        "base_z": float(base_z),
        "wall_quads": wall_quads,
        "full_model_meshes": full_model_meshes,
        "model_boundary_edges_xyz": _model_boundary_edges_xyz_by_class(
            edge_groups,
            corners,
            id_to_idx,
        ),
        "footprint": footprint,
        "source_crs": str(SOURCE_CRS),
    }


# Compatibility name retained for the standalone experiment.
def target_wall_meshes(model_geometry, wall_indices: Sequence[int]):
    meshes = []
    quads = []
    wall_quads = model_geometry["wall_quads"]
    for wall_index in wall_indices:
        quad = np.asarray(wall_quads.get(int(wall_index)), dtype=np.float64)
        if not _finite_quad(quad):
            continue
        quads.append(quad)
        meshes.append(_quad_mesh(quad, f"target_wall_{int(wall_index):02d}"))
    if not meshes:
        raise ValueError("No valid target-wall geometry was found for the facade group.")
    return meshes, quads


def candidate_camera_pose(candidate: Mapping[str, object], image_size: str):
    camera_xyz = np.asarray(candidate["camera_utm_xyz"], dtype=np.float64)
    projection_heading = float(candidate.get(
        "projection_heading_deg",
        candidate.get("heading_deg", 0.0),
    ))
    pitch = float(candidate.get("pitch_deg", 0.0))
    fov = float(candidate.get("fov_deg", 100.0))
    K, R_wc, C = build_pose_from_heading_pitch(
        camera_xyz,
        projection_heading,
        pitch,
        img_size=image_size,
        fov_deg=fov,
    )
    return K, R_wc, C


def _outline_points_from_depth(depth_map):
    mask = np.isfinite(depth_map) & (np.asarray(depth_map) > 0.0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [contour for contour in contours if len(contour) >= 3]
    if not contours:
        raise ValueError("The target wall has no projected image-space outline.")
    contour = max(contours, key=lambda value: abs(float(cv2.contourArea(value))))
    return np.asarray(contour[:, 0, :], dtype=np.float64)


def _alignment_summary(
    *,
    homography,
    applied,
    source,
    reason,
    transform=None,
    score_improvement=0.0,
    fit_geometry_source=None,
    semantic_error=None,
):
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Depth-global alignment homography must be a finite 3x3 matrix.")
    if abs(float(np.linalg.det(H))) < 1.0e-12:
        raise ValueError("Depth-global alignment homography is singular.")
    transform = dict(transform or {})
    return {
        "homography": H,
        "applied": bool(applied),
        "effective_mode": "depth_global" if applied else "raw_fallback",
        "alignment_source": str(source),
        "reason": str(reason),
        "scale": float(transform.get("scale", np.hypot(H[0, 0], H[1, 0]))),
        "rotation_deg": float(transform.get("rotation_deg", 0.0)),
        "tx_px": float(transform.get("tx", H[0, 2])),
        "ty_px": float(transform.get("ty", H[1, 2])),
        "score_improvement": float(score_improvement),
        "fit_geometry_source": fit_geometry_source,
        "semantic_guide_error": semantic_error,
    }


def fit_candidate_depth_global_alignment(
    *,
    candidate: Mapping[str, object],
    source_image_rgb: np.ndarray,
    raw_target_depth: np.ndarray,
    full_model_meshes,
    model_boundary_edges_xyz,
    image_size: str,
    existing_alignment: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return the accepted production-equivalent global fit for one candidate."""
    if existing_alignment and bool(existing_alignment.get("applied", False)):
        return _alignment_summary(
            homography=existing_alignment["homography"],
            applied=True,
            source=existing_alignment.get("alignment_source", "production_metadata"),
            reason=existing_alignment.get("reason", "accepted_production_depth_global_fit"),
            transform=existing_alignment.get("transform"),
            score_improvement=existing_alignment.get("score_improvement", 0.0),
            fit_geometry_source=existing_alignment.get("fit_geometry_source"),
        )

    K, R_wc, C = candidate_camera_pose(candidate, image_size)
    width, height = [int(value) for value in image_size.lower().split("x")]
    full_depth = render_model_depth_map(
        full_model_meshes,
        K,
        R_wc,
        C,
        (width, height),
    )
    if not np.any(np.isfinite(full_depth) & (full_depth > 0.0)):
        return _alignment_summary(
            homography=np.eye(3),
            applied=False,
            source="experiment_recomputed",
            reason="whole_model_not_visible",
        )

    semantic_geometry = None
    semantic_error = None
    if bool(getattr(pipeline_config, "MODEL_DEPTH_BOUNDARY_USE_SEMANTIC_GUIDES", True)):
        try:
            semantic_geometry = project_semantic_model_boundary_edges(
                model_edges_xyz_by_class=model_boundary_edges_xyz or {},
                K=K,
                R_wc=R_wc,
                C=C,
                full_model_depth=full_depth,
                image_to_output_H=np.eye(3, dtype=np.float64),
                near_m=float(getattr(pipeline_config, "MODEL_DEPTH_NEAR_M", 0.05)),
                sample_step_px=float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_SAMPLE_STEP_PX",
                    2.0,
                )),
                silhouette_tolerance_px=float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_SILHOUETTE_TOLERANCE_PX",
                    4.0,
                )),
                depth_search_radius_px=int(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_SEARCH_RADIUS_PX",
                    2,
                )),
                depth_tolerance_m=float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_TOLERANCE_M",
                    0.35,
                )),
                depth_relative_tolerance=float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_RELATIVE_TOLERANCE",
                    0.03,
                )),
                maximum_visibility_gap_samples=int(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_MAX_GAP_SAMPLES",
                    2,
                )),
                minimum_visible_run_px=float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_MIN_RUN_PX",
                    8.0,
                )),
            )
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"

    raw_outline = _outline_points_from_depth(raw_target_depth)
    fit_config = make_production_fit_config(
        allow_rotation=bool(getattr(
            pipeline_config,
            "MODEL_DEPTH_BOUNDARY_FIT_ALLOW_ROTATION",
            False,
        )),
        minimum_score_improvement=float(getattr(
            pipeline_config,
            "MODEL_DEPTH_BOUNDARY_FIT_MIN_SCORE_IMPROVEMENT",
            0.025,
        )),
    )
    try:
        fit = fit_depth_silhouette_to_image(
            image_bgr=cv2.cvtColor(
                np.asarray(source_image_rgb, dtype=np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
            full_model_depth=full_depth,
            raw_wall_outline_px=raw_outline,
            wall_local_fit_outline_px=raw_outline,
            fit_config=fit_config,
            minimum_area_px=int(getattr(
                pipeline_config,
                "MODEL_DEPTH_BOUNDARY_FIT_MIN_AREA_PX",
                350,
            )),
            minimum_component_fraction=float(getattr(
                pipeline_config,
                "MODEL_DEPTH_BOUNDARY_FIT_MIN_COMPONENT_FRACTION",
                0.02,
            )),
            contour_epsilon_px=float(getattr(
                pipeline_config,
                "MODEL_DEPTH_BOUNDARY_FIT_CONTOUR_EPSILON_PX",
                1.5,
            )),
            maximum_points=int(getattr(
                pipeline_config,
                "MODEL_DEPTH_BOUNDARY_FIT_MAX_POINTS",
                240,
            )),
            semantic_boundary_geometry=semantic_geometry,
            semantic_class_weights={
                "roof": float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_ROOF_WEIGHT",
                    3.0,
                )),
                "wall": float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_WALL_WEIGHT",
                    2.0,
                )),
                "base": float(getattr(
                    pipeline_config,
                    "MODEL_DEPTH_BOUNDARY_BASE_WEIGHT",
                    0.35,
                )),
            },
        )
    except Exception as exc:
        return _alignment_summary(
            homography=np.eye(3),
            applied=False,
            source="experiment_recomputed",
            reason=f"depth_global_fit_failed: {type(exc).__name__}: {exc}",
            semantic_error=semantic_error,
        )

    applied = bool(fit.get("applied", False))
    return _alignment_summary(
        homography=(fit["homography"] if applied else np.eye(3)),
        applied=applied,
        source="experiment_recomputed",
        reason=str(fit.get("reason", "accepted" if applied else "not_accepted")),
        transform=fit.get("transform"),
        score_improvement=fit.get("score_improvement", 0.0),
        fit_geometry_source=fit.get("fit_geometry_source"),
        semantic_error=semantic_error,
    )


def _parse_numeric(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    if "ft" in text or "feet" in text or "'" in text:
        number *= 0.3048
    return number if math.isfinite(number) else None


def estimate_osm_building_height(
    tags: Mapping[str, str],
    *,
    default_height_m: float = 15.0,
    level_height_m: float = 3.0,
) -> Tuple[float, float]:
    """Estimate total and minimum heights from standard OSM building tags."""
    height = _parse_numeric(tags.get("height"))
    if height is None:
        levels = _parse_numeric(tags.get("building:levels"))
        roof_height = _parse_numeric(tags.get("roof:height"))
        roof_levels = _parse_numeric(tags.get("roof:levels"))
        if levels is not None:
            height = levels * float(level_height_m)
            if roof_height is not None:
                height += roof_height
            elif roof_levels is not None:
                height += roof_levels * float(level_height_m)
    if height is None or height <= 0.0:
        height = float(default_height_m)

    min_height = _parse_numeric(tags.get("min_height"))
    if min_height is None:
        min_level = _parse_numeric(tags.get("building:min_level"))
        min_height = 0.0 if min_level is None else min_level * float(level_height_m)
    min_height = float(np.clip(min_height, 0.0, max(height - 0.25, 0.0)))
    return float(np.clip(height, 1.0, 150.0)), min_height


def _way_polygon(element):
    geometry = element.get("geometry") or []
    coordinates = [
        (float(point["lon"]), float(point["lat"]))
        for point in geometry
        if "lon" in point and "lat" in point
    ]
    if len(coordinates) < 3:
        return None
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    polygon = Polygon(coordinates)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon if not polygon.is_empty else None


def _relation_polygon(element):
    outer_lines = []
    inner_lines = []
    for member in element.get("members", []):
        geometry = member.get("geometry") or []
        coordinates = [
            (float(point["lon"]), float(point["lat"]))
            for point in geometry
            if "lon" in point and "lat" in point
        ]
        if len(coordinates) < 2:
            continue
        line = LineString(coordinates)
        if str(member.get("role", "outer")) == "inner":
            inner_lines.append(line)
        else:
            outer_lines.append(line)
    if not outer_lines:
        return None
    outers = list(polygonize(unary_union(outer_lines)))
    if not outers:
        return None
    polygon = unary_union(outers)
    if inner_lines:
        inners = list(polygonize(unary_union(inner_lines)))
        if inners:
            polygon = polygon.difference(unary_union(inners))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon if not polygon.is_empty else None


def parse_overpass_buildings(
    payload: Mapping[str, object],
    *,
    target_crs: str = SOURCE_CRS,
    default_height_m: float = 15.0,
    level_height_m: float = 3.0,
) -> List[OSMBuilding]:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    buildings = []
    seen = set()
    for element in payload.get("elements", []):
        osm_type = str(element.get("type", ""))
        osm_id = int(element.get("id", -1))
        key = (osm_type, osm_id)
        if key in seen or osm_id < 0:
            continue
        polygon_wgs84 = (
            _way_polygon(element)
            if osm_type == "way"
            else _relation_polygon(element)
            if osm_type == "relation"
            else None
        )
        if polygon_wgs84 is None:
            continue
        projected = shapely_transform(transformer.transform, polygon_wgs84)
        polygon_parts = (
            list(projected.geoms)
            if projected.geom_type == "MultiPolygon"
            else [projected]
        )
        tags = {str(k): str(v) for k, v in dict(element.get("tags", {})).items()}
        height, min_height = estimate_osm_building_height(
            tags,
            default_height_m=default_height_m,
            level_height_m=level_height_m,
        )
        for part_index, polygon in enumerate(polygon_parts):
            if polygon.is_empty or polygon.area < 1.0:
                continue
            buildings.append(OSMBuilding(
                osm_type=osm_type,
                osm_id=osm_id,
                footprint=polygon,
                tags=tags,
                height_m=height,
                min_height_m=min_height,
                part_index=int(part_index),
            ))
        seen.add(key)
    return buildings


def fetch_osm_buildings(
    *,
    model_footprint,
    source_crs: str = SOURCE_CRS,
    radius_m: float = 120.0,
    endpoint: str = DEFAULT_OVERPASS_ENDPOINT,
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    timeout_s: float = 90.0,
    default_height_m: float = 15.0,
    level_height_m: float = 3.0,
) -> Tuple[List[OSMBuilding], Dict[str, object]]:
    """Fetch nearby OSM building footprints with a deterministic local cache."""
    center = model_footprint.centroid
    to_wgs84 = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    longitude, latitude = to_wgs84.transform(float(center.x), float(center.y))
    latitude_padding = float(radius_m) / 111_320.0
    longitude_padding = float(radius_m) / max(
        111_320.0 * math.cos(math.radians(float(latitude))),
        1.0,
    )
    south = float(latitude) - latitude_padding
    north = float(latitude) + latitude_padding
    west = float(longitude) - longitude_padding
    east = float(longitude) + longitude_padding
    bbox = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
    query = (
        "[out:json][timeout:35];"
        "("
        f'way["building"]({bbox});'
        f'relation["building"]({bbox});'
        ");out tags geom;"
    )
    cache_payload = {
        "endpoint": endpoint,
        "latitude": round(float(latitude), 7),
        "longitude": round(float(longitude), 7),
        "radius_m": round(float(radius_m), 1),
        "bbox": bbox,
        "query": query,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = None
    payload = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"overpass_{cache_key}.json"
        if cache_path.exists() and not refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
    used_endpoint = endpoint
    if payload is None:
        endpoints = [endpoint]
        if endpoint == DEFAULT_OVERPASS_ENDPOINT:
            endpoints.extend(
                fallback for fallback in OVERPASS_FALLBACK_ENDPOINTS
                if fallback not in endpoints
            )
        errors = []
        headers = {
            "User-Agent": OVERPASS_USER_AGENT,
            "Accept": "application/json",
        }
        for candidate_endpoint in endpoints:
            try:
                response = requests.get(
                    candidate_endpoint,
                    params={"data": query},
                    headers=headers,
                    timeout=float(timeout_s),
                )
                response.raise_for_status()
                payload = response.json()
                used_endpoint = candidate_endpoint
                break
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{candidate_endpoint}: {type(exc).__name__}: {exc}")
        if payload is None:
            raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))
        if cache_path is not None:
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
    buildings = parse_overpass_buildings(
        payload,
        target_crs=source_crs,
        default_height_m=default_height_m,
        level_height_m=level_height_m,
    )
    metadata = {
        "endpoint": used_endpoint,
        "query_center_wgs84": [float(latitude), float(longitude)],
        "query_bbox_wgs84": [south, west, north, east],
        "radius_m": float(radius_m),
        "cache_path": str(cache_path) if cache_path is not None else None,
        "raw_element_count": int(len(payload.get("elements", []))),
        "parsed_building_count": int(len(buildings)),
        "attribution": "OpenStreetMap contributors",
        "license": "ODbL 1.0",
    }
    return buildings, metadata


def remove_target_osm_buildings(
    buildings: Sequence[OSMBuilding],
    model_footprint,
    *,
    minimum_overlap_fraction: float = 0.35,
    centroid_tolerance_m: float = 2.0,
) -> Tuple[List[OSMBuilding], List[str]]:
    """Remove OSM footprints representing the LoD-2 building itself."""
    blockers = []
    excluded = []
    target_buffer = model_footprint.buffer(float(centroid_tolerance_m))
    for building in buildings:
        footprint = building.footprint
        intersection_area = float(footprint.intersection(model_footprint).area)
        overlap = intersection_area / max(min(float(footprint.area), float(model_footprint.area)), 1.0e-6)
        centroid_matches = bool(
            target_buffer.contains(footprint.centroid)
            and intersection_area / max(float(footprint.area), 1.0e-6) >= 0.20
        )
        if overlap >= float(minimum_overlap_fraction) or centroid_matches:
            excluded.append(building.key)
        else:
            blockers.append(building)
    return blockers, excluded


def _polygon_prism_mesh(polygon, bottom_z, top_z):
    vertices = []
    faces = []

    def add_triangle(points_xyz):
        start = len(vertices)
        vertices.extend(points_xyz)
        faces.append([start, start + 1, start + 2])

    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        coordinates = list(triangle.exterior.coords)[:3]
        add_triangle([(x, y, top_z) for x, y in coordinates])
        add_triangle([(x, y, bottom_z) for x, y in reversed(coordinates)])

    rings = [polygon.exterior, *list(polygon.interiors)]
    for ring in rings:
        coordinates = list(ring.coords)
        for point0, point1 in zip(coordinates[:-1], coordinates[1:]):
            x0, y0 = point0
            x1, y1 = point1
            start = len(vertices)
            vertices.extend([
                (x0, y0, bottom_z),
                (x1, y1, bottom_z),
                (x1, y1, top_z),
                (x0, y0, top_z),
            ])
            faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
    if not faces:
        return None
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def build_osm_blocker_meshes(
    buildings: Sequence[OSMBuilding],
    *,
    ground_z: float,
) -> Tuple[List[Tuple[str, trimesh.Trimesh]], Dict[str, OSMBuilding]]:
    meshes = []
    lookup = {}
    for building in buildings:
        bottom_z = float(ground_z) + float(building.min_height_m)
        top_z = float(ground_z) + float(building.height_m)
        mesh = _polygon_prism_mesh(building.footprint, bottom_z, top_z)
        if mesh is None:
            continue
        name = f"osm_{building.osm_type}_{building.osm_id}_part_{building.part_index}"
        meshes.append((name, mesh))
        lookup[name] = building
    return meshes, lookup


def _candidate_blocker_names(
    camera_xyz,
    target_quads,
    blocker_lookup,
    *,
    corridor_buffer_m: float = 1.0,
):
    camera_point = Point(float(camera_xyz[0]), float(camera_xyz[1]))
    target_points = np.vstack(target_quads)[:, :2]
    target_hull = MultiPoint(target_points).convex_hull
    corridor = unary_union([camera_point, target_hull]).convex_hull.buffer(
        float(corridor_buffer_m)
    )
    target_distance = max(
        float(camera_point.distance(Point(float(x), float(y))))
        for x, y in target_points
    )
    names = []
    for name, building in blocker_lookup.items():
        footprint = building.footprint
        if not footprint.intersects(corridor):
            continue
        if float(camera_point.distance(footprint)) >= target_distance + float(corridor_buffer_m):
            continue
        names.append(name)
    return names


def evaluate_candidate_occlusion(
    *,
    candidate: Mapping[str, object],
    target_meshes,
    target_quads,
    blocker_meshes,
    blocker_lookup,
    image_size: str = "640x640",
    depth_tolerance_m: float = 0.10,
    corridor_buffer_m: float = 1.0,
    target_alignment_H: Optional[np.ndarray] = None,
    precomputed_raw_target_depth: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Measure OSM occlusion inside the corrected depth-global wall projection."""
    K, R_wc, C = candidate_camera_pose(candidate, image_size)
    camera_xyz = np.asarray(C, dtype=np.float64)
    width, height = [int(value) for value in image_size.lower().split("x")]
    raw_target_depth = (
        np.asarray(precomputed_raw_target_depth, dtype=np.float32)
        if precomputed_raw_target_depth is not None
        else render_model_depth_map(
            target_meshes,
            K,
            R_wc,
            C,
            (width, height),
        )
    )
    if raw_target_depth.shape != (height, width):
        raise ValueError("Precomputed target depth does not match the candidate image size.")
    alignment_H = np.asarray(
        np.eye(3) if target_alignment_H is None else target_alignment_H,
        dtype=np.float64,
    )
    target_depth = warp_depth_map_to_canvas(
        raw_target_depth,
        alignment_H,
        (width, height),
    )
    raw_target_mask = np.isfinite(raw_target_depth) & (raw_target_depth > 0.0)
    target_mask = np.isfinite(target_depth) & (target_depth > 0.0)
    blocker_names = _candidate_blocker_names(
        camera_xyz,
        target_quads,
        blocker_lookup,
        corridor_buffer_m=corridor_buffer_m,
    )
    blocker_name_set = set(blocker_names)
    candidate_meshes = [
        (name, mesh)
        for name, mesh in blocker_meshes
        if name in blocker_name_set
    ]
    blocker_depth = (
        render_model_depth_map(candidate_meshes, K, R_wc, C, (width, height))
        if candidate_meshes
        else np.full((height, width), np.nan, dtype=np.float32)
    )
    occlusion_mask = (
        target_mask
        & np.isfinite(blocker_depth)
        & (blocker_depth + float(depth_tolerance_m) < target_depth)
    )
    target_pixels = int(target_mask.sum())
    occluded_pixels = int(occlusion_mask.sum())
    occluded_fraction = (
        float(occluded_pixels / target_pixels)
        if target_pixels > 0 else 1.0
    )
    return {
        "raw_target_depth": raw_target_depth,
        "raw_target_mask": raw_target_mask,
        "target_depth": target_depth,
        "blocker_depth": blocker_depth,
        "target_mask": target_mask,
        "occlusion_mask": occlusion_mask,
        "target_pixel_count": target_pixels,
        "osm_occluded_pixel_count": occluded_pixels,
        "osm_occluded_fraction": occluded_fraction,
        "osm_visible_fraction": float(max(0.0, 1.0 - occluded_fraction)),
        "osm_fully_clear": bool(occluded_pixels == 0),
        "candidate_blocker_mesh_names": blocker_names,
        "target_alignment_H": alignment_H,
        "K": K,
        "R_wc": R_wc,
        "C": C,
    }


def select_candidate_with_osm_visibility(
    evaluations: Sequence[Mapping[str, object]],
    *,
    clear_occlusion_fraction: float = 0.005,
) -> Dict[str, object]:
    """Select the camera with the greatest net visible target-wall fraction."""
    usable = [row for row in evaluations if int(row.get("target_pixel_count", 0)) > 0]
    if not usable:
        raise ValueError("No candidate projects the target wall into the image.")
    clear = [
        row for row in usable
        if float(row.get("osm_occluded_fraction", 1.0)) <= float(clear_occlusion_fraction)
    ]

    def selection_key(row):
        candidate = row.get("candidate", {})
        raw_visibility = float(np.clip(
            candidate.get(
                "target_usable_visibility_fraction",
                candidate.get("projected_coverage_fraction", 0.0),
            ),
            0.0,
            1.0,
        ))
        occluded_fraction = float(np.clip(
            row.get("osm_occluded_fraction", 1.0),
            0.0,
            1.0,
        ))
        net_visibility = raw_visibility * (1.0 - occluded_fraction)
        return (
            net_visibility,
            -occluded_fraction,
            raw_visibility,
            float(candidate.get("projected_coverage_fraction", 0.0)),
            -int(candidate.get("source_selection_rank", 10**6)),
            -int(candidate.get("source_index", 10**6)),
        )

    selected = max(usable, key=selection_key)
    selected_occlusion = float(selected.get("osm_occluded_fraction", 1.0))
    fallback_mask_required = bool(
        selected_occlusion > float(clear_occlusion_fraction)
    )
    reason = (
        "maximum_net_target_visibility_with_osm_removal"
        if fallback_mask_required
        else "maximum_net_target_visibility"
    )
    return {
        "selected": selected,
        "selection_reason": reason,
        "fallback_mask_required": fallback_mask_required,
        "clear_candidate_count": int(len(clear)),
        "usable_candidate_count": int(len(usable)),
        "clear_occlusion_fraction": float(clear_occlusion_fraction),
    }


def mask_outline(mask):
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours
