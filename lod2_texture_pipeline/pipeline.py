# -*- coding: utf-8 -*-
"""Batch pipeline orchestration for LoD-2 facade and roof texturing."""

import json
import math
import os
import re
import shutil
import struct
import tempfile
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np
import rasterio
import torch
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import Point

from .config import *
from .diagnostic_overlay_style import (
    ACCEPTED_MODEL_LINE,
    BACKGROUND_AWARE_SEMANTIC_LEGEND_ROWS,
    OSM_LEGEND_ROW,
    OSM_OBSTRUCTION_LINE,
    RAW_MODEL_LINE,
    SEARCH_LEGEND_ROW,
    SEMANTIC_LEGEND_ROWS,
    STRICT_ROOF_AUDIT_LEGEND_ROW,
    OverlayLineStyle,
    draw_legend,
    draw_styled_line,
    model_projection_legend,
)
from .depth_boundary_fit import (
    create_depth_boundary_fit_overlay,
    create_depth_silhouette_shift_overlay,
    depth_boundary_fit_metadata,
    extract_depth_silhouette_geometry,
    filter_image_border_wrapper_segments,
    fit_depth_silhouette_to_image,
    project_semantic_model_boundary_edges,
)
from .facade_alignment import facade_alignment_metadata, select_facade_alignment
from .facade_refinement import (
    build_post_hough_roof_structure_removal,
    build_reused_prefit_facade_mask,
)
from .facade_side_evidence import (
    analyze_source_side_evidence,
    build_adjacent_wall_contexts,
    side_evidence_metadata,
    warp_side_evidence_to_rectified,
)
from .depth_aware_region_fit import (
    DepthAwareRegionFitConfig,
    create_depth_aware_region_fit_overlay,
    depth_aware_region_fit_metadata,
    fit_depth_aware_segmentation_region,
    visible_group_mask_from_depth,
)
from .dgm_elevation import (
    CameraElevationResolver,
    InMemoryThuringiaDGM1,
    unique_base_vertices_from_edges,
)
from .geojson_io import build_edge_loops_from_gdf, load_3d_geojson
from .inpainting import (
    bleed_rgb_into_transparency,
    build_wall_region_mask,
    lama_fill_rectified_wall,
)
from .mesh import (
    _build_wall_mesh_from_verts,
    build_closed_roof_polygons,
    build_trimesh_from_surface_face,
    rasterize_polygons_to_mask,
    repair_mesh_t_junctions,
    triangulate_surface,
)
from .osm_occlusion import (
    DEFAULT_OVERPASS_ENDPOINT,
    build_model_occlusion_geometry,
    build_osm_blocker_meshes,
    evaluate_candidate_occlusion,
    fetch_osm_buildings,
    remove_target_osm_buildings,
)
from .projection import *
from .projection import _closed_polyline_self_intersects
from .prefit_semantic_guidance import (
    PrefitSemanticGuidanceConfig,
    assess_prefit_candidate_visibility,
    build_prefit_semantic_guidance,
    create_prefit_semantic_guidance_overlay,
)
from .posttexture_base_repair import level_finished_building_base
from .opening_rectification import (
    estimate_opening_aware_rectification,
    run_opening_sam3_prompts,
)
from .quadfit import *
from .streetview import *
from .utils import _mask_key, ensure_outdir, name_for, save_sam3_instance_debug_overlay, save_viewer_bundle_npz, save_with_overlay
from .wireframe_fit import (
    apply_homography as apply_H,
    make_production_fit_config,
    semantic_boundary_alignment_score,
)

class _NoopStage:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _TimedStage:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = str(name)
        self.started = None

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.started
        self.timer.record(self.name, elapsed)
        return False


class _PipelineTimer:
    def __init__(self, label):
        self.label = str(label)
        self.started = time.perf_counter()
        self.events = []
        self.finished = False

    def stage(self, name):
        return _TimedStage(self, name)

    def record(self, name, elapsed):
        elapsed = float(elapsed)
        self.events.append({"stage": str(name), "seconds": elapsed})
        print(f"[time] {self.label} | {name}: {elapsed:.2f}s")

    def finish(self, out_dir=None):
        if self.finished:
            return
        self.finished = True
        total = time.perf_counter() - self.started
        print(f"[time] {self.label} | TOTAL: {total:.2f}s")
        if self.events:
            print(f"[time] {self.label} | slowest stages:")
            for row in sorted(self.events, key=lambda r: r["seconds"], reverse=True)[:12]:
                print(f"        {row['seconds']:8.2f}s  {row['stage']}")
        if out_dir is not None:
            try:
                path = Path(out_dir) / "stage_timings.json"
                payload = {
                    "label": self.label,
                    "total_seconds": float(total),
                    "events": self.events,
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"[time] {self.label} | saved timings: {path}")
            except Exception as exc:
                print(f"[time] {self.label} | failed to save timing JSON: {exc}")


def _timer_stage(timer, name):
    return timer.stage(name) if timer is not None else _NoopStage()


def _build_dgm_camera_elevation_resolver(
    building_label,
    *,
    corners,
    base_edges,
    id_to_idx,
    base_z,
):
    enabled = bool(globals().get("ENABLE_DGM_CAMERA_ELEVATION", True))
    base_vertices = unique_base_vertices_from_edges(
        corners,
        base_edges,
        id_to_idx,
    )
    sampler = None
    if enabled:
        sampler = InMemoryThuringiaDGM1(
            url_template=str(globals().get(
                "DGM1_TILE_URL_TEMPLATE",
                (
                    "https://geoportal.geoportal-th.de/hoehendaten/DGM/"
                    "dgm_2020-2025/"
                    "dgm1_32_{easting_km}_{northing_km}_1_th_2020-2025.zip"
                ),
            )),
            timeout_seconds=float(globals().get("DGM1_HTTP_TIMEOUT_S", 30.0)),
            max_memory_tiles=int(globals().get("DGM1_MAX_MEMORY_TILES", 4)),
            expected_horizontal_epsg=int(globals().get(
                "DGM1_EXPECTED_HORIZONTAL_EPSG", 25832,
            )),
            expected_vertical_epsg=int(globals().get(
                "DGM1_EXPECTED_VERTICAL_EPSG", 7837,
            )),
        )

    return CameraElevationResolver(
        building_label=str(building_label),
        sampler=sampler,
        base_vertices=base_vertices,
        fallback_base_z=float(base_z),
        camera_height_m=float(FIXED_HEIGHT_M),
        enabled=enabled,
        minimum_inlier_vertices=int(globals().get(
            "DGM_BASE_MIN_INLIER_VERTICES", 3,
        )),
        minimum_inlier_fraction=float(globals().get(
            "DGM_BASE_MIN_INLIER_FRACTION", 0.66,
        )),
        outlier_mad_scale=float(globals().get(
            "DGM_BASE_OUTLIER_MAD_SCALE", 3.5,
        )),
        outlier_minimum_deviation_m=float(globals().get(
            "DGM_BASE_OUTLIER_MIN_DEVIATION_M", 0.50,
        )),
        maximum_inlier_absolute_difference_m=float(globals().get(
            "DGM_BASE_MAX_INLIER_ABS_DIFFERENCE_M", 0.75,
        )),
        maximum_median_absolute_difference_m=float(globals().get(
            "DGM_BASE_MAX_MEDIAN_ABS_DIFFERENCE_M", 0.50,
        )),
    )


def patch_glb_materials_double_sided(glb_path, asset_extras=None) -> bool:
    """
    Patch exported GLB materials for reliable, matte texture rendering.

    Trimesh does not expose these settings consistently for every generated
    material type.  Keep structural roof and wall triangles opaque: their PNG
    alpha is only a texture-generation mask and must not cut holes in the mesh.
    Restore Trimesh's established photo multiplier, remove metallic reflections,
    and clamp sampling at texture boundaries.
    """
    try:
        with open(glb_path, "rb") as f:
            data = f.read()
    except OSError:
        return False

    if len(data) < 20:
        return False

    magic, version, _total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        return False

    chunks = []
    offset = 12
    json_chunk_index = None

    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset:offset + chunk_len]
        offset += chunk_len
        chunks.append([chunk_type, chunk_data])
        if chunk_type == 0x4E4F534A and json_chunk_index is None:
            json_chunk_index = len(chunks) - 1

    if json_chunk_index is None:
        return False

    try:
        gltf = json.loads(chunks[json_chunk_index][1].rstrip(b" \t\r\n\0").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False

    changed = False

    if asset_extras:
        asset = gltf.setdefault("asset", {})
        extras = asset.setdefault("extras", {})
        for key, value in asset_extras.items():
            if extras.get(key) != value:
                extras[key] = value
                changed = True

    def default_double_sided_material():
        return {
            "name": "default_double_sided",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        }

    materials = gltf.setdefault("materials", [])
    default_material = None
    if len(materials) == 0:
        materials.append(default_double_sided_material())
        default_material = 0
        changed = True

    base_color_texture_indices = set()
    for material in materials:
        if material.get("doubleSided") is not True:
            material["doubleSided"] = True
            changed = True

        pbr = material.get("pbrMetallicRoughness")
        if (
            material.get("name") == "default_double_sided"
            and isinstance(pbr, dict)
        ):
            if pbr.get("metallicFactor") != 0.0:
                pbr["metallicFactor"] = 0.0
                changed = True
            if pbr.get("roughnessFactor") != 1.0:
                pbr["roughnessFactor"] = 1.0
                changed = True
        if isinstance(pbr, dict) and "baseColorTexture" in pbr:
            base_color_texture = pbr.get("baseColorTexture")
            if isinstance(base_color_texture, dict):
                texture_index = base_color_texture.get("index")
                if isinstance(texture_index, int) and texture_index >= 0:
                    base_color_texture_indices.add(texture_index)
            # Repair files produced by the short-lived MASK + factor-1 patch.
            # Fresh Trimesh exports already carry the established 0.4 photo
            # multiplier; otherwise preserve any explicitly authored factor.
            regressed_photo_material = (
                pbr.get("baseColorFactor") == [1.0, 1.0, 1.0, 1.0]
                and material.get("alphaMode") == "MASK"
                and material.get("alphaCutoff") == 0.5
            )
            if regressed_photo_material:
                pbr["baseColorFactor"] = [0.4, 0.4, 0.4, 1.0]
                changed = True
            if pbr.get("metallicFactor") != 0.0:
                pbr["metallicFactor"] = 0.0
                changed = True
            if pbr.get("roughnessFactor") != 1.0:
                pbr["roughnessFactor"] = 1.0
                changed = True
            if material.get("alphaMode") != "OPAQUE":
                material["alphaMode"] = "OPAQUE"
                changed = True
            if "alphaCutoff" in material:
                del material["alphaCutoff"]
                changed = True

    textures = gltf.get("textures", [])
    valid_base_color_texture_indices = {
        index for index in base_color_texture_indices
        if index < len(textures)
    }
    if valid_base_color_texture_indices:
        samplers = gltf.setdefault("samplers", [])
        clamp_sampler = {
            "magFilter": 9729,   # LINEAR
            # Avoid averaging distant transparent/black RGB into the facade
            # edge at lower mip levels.  The exported textures already carry a
            # finite RGB gutter around their structural UV footprint.
            "minFilter": 9729,   # LINEAR (no mipmaps)
            "wrapS": 33071,      # CLAMP_TO_EDGE
            "wrapT": 33071,
        }
        clamp_sampler_index = next(
            (
                index for index, sampler in enumerate(samplers)
                if all(sampler.get(key) == value
                       for key, value in clamp_sampler.items())
            ),
            None,
        )
        if clamp_sampler_index is None:
            samplers.append(clamp_sampler)
            clamp_sampler_index = len(samplers) - 1
            changed = True
        for texture_index in sorted(valid_base_color_texture_indices):
            texture = textures[texture_index]
            if texture.get("sampler") != clamp_sampler_index:
                texture["sampler"] = clamp_sampler_index
                changed = True

    for mesh_def in gltf.get("meshes", []):
        for primitive in mesh_def.get("primitives", []):
            if "material" not in primitive:
                if default_material is None:
                    materials.append(default_double_sided_material())
                    default_material = len(materials) - 1
                    changed = True
                primitive["material"] = default_material
                changed = True

    if not changed:
        return False

    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - (len(json_bytes) % 4)) % 4)
    chunks[json_chunk_index][1] = json_bytes

    total_len = 12 + sum(8 + len(chunk_data) for _, chunk_data in chunks)
    out = bytearray(struct.pack("<III", magic, version, total_len))
    for chunk_type, chunk_data in chunks:
        out.extend(struct.pack("<II", len(chunk_data), chunk_type))
        out.extend(chunk_data)

    try:
        with open(glb_path, "wb") as f:
            f.write(out)
    except OSError:
        return False

    return True

def _unit_xy(v):
    arr = np.asarray(v, dtype=float)[:2]
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        return np.zeros(2, dtype=float)
    return arr / n

def _wall_quad_from_edge_pair(corners, id_to_idx, edge_a, edge_b):
    s1, t1 = edge_a
    s2, t2 = edge_b
    if any(nid not in id_to_idx for nid in [s1, t1, s2, t2]):
        return None
    p1a = corners[id_to_idx[s1]]
    p1b = corners[id_to_idx[t1]]
    p2a = corners[id_to_idx[s2]]
    p2b = corners[id_to_idx[t2]]

    def by_z(a, b):
        return (a, b) if a[2] <= b[2] else (b, a)

    b1, t1p = by_z(p1a, p1b)
    b2, t2p = by_z(p2a, p2b)
    return np.vstack([b1, b2, t2p, t1p]).astype(np.float64)

def _epsg25832_vertices_to_gltf_y_up(vertices_epsg, origin_epsg):
    """
    Source coordinates are EPSG meters with X=east, Y=north, Z=up.
    glTF/COLLADA viewers expect Y-up local model coordinates:
    X=east, Y=up, Z=-north.
    """
    local = np.asarray(vertices_epsg, dtype=np.float64) - np.asarray(origin_epsg, dtype=np.float64)
    return np.column_stack([local[:, 0], local[:, 2], -local[:, 1]])

def _all_mesh_vertices(meshes_named):
    vertices = []
    for _name, mesh in meshes_named:
        v = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
        if v.ndim == 2 and v.shape[1] == 3 and v.shape[0] > 0:
            finite = np.isfinite(v).all(axis=1)
            if finite.any():
                vertices.append(v[finite])
    return np.vstack(vertices) if vertices else None

def _make_export_origin(meshes_named, relative_to_ground=False):
    vertices = _all_mesh_vertices(meshes_named)
    if vertices is None:
        return None
    origin = np.nanmean(vertices, axis=0)
    if relative_to_ground:
        origin[2] = float(np.nanmin(vertices[:, 2]))
    return origin.astype(np.float64)

def _copy_mesh_for_local_y_up(mesh, origin_epsg):
    out = mesh.copy()
    out.vertices = _epsg25832_vertices_to_gltf_y_up(out.vertices, origin_epsg)
    return out

def _copy_meshes_for_local_y_up(meshes_named, origin_epsg):
    return [(name, _copy_mesh_for_local_y_up(mesh, origin_epsg)) for name, mesh in meshes_named]

def _xml_escape(text):
    text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

def _safe_collada_id(text):
    text = str(text)
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)
    if not safe or safe[0].isdigit():
        safe = "id_" + safe
    return safe

def _collada_float_list(values):
    return " ".join(f"{float(v):.6f}" for v in np.asarray(values).reshape(-1))

def _mesh_texture_image(mesh):
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    image = None
    if material is not None:
        image = getattr(material, "image", None)
        if image is None:
            image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(visual, "image", None)
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    arr = np.asarray(image)
    if arr.ndim in (2, 3):
        return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    return None

def _mesh_uv_array(mesh):
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != len(mesh.vertices):
        return None
    if not np.isfinite(uv).all():
        return None
    return uv

def _mesh_face_colors(mesh):
    visual = getattr(mesh, "visual", None)
    colors = getattr(visual, "face_colors", None)
    if colors is None or len(colors) == 0:
        return np.tile(np.array([[220, 220, 220, 255]], dtype=np.uint8), (len(mesh.faces), 1))
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.ndim == 1:
        colors = np.tile(colors.reshape(1, -1), (len(mesh.faces), 1))
    if colors.shape[1] == 3:
        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        colors = np.hstack([colors, alpha])
    if colors.shape[0] != len(mesh.faces):
        colors = np.tile(colors[0:1, :4], (len(mesh.faces), 1))
    return colors[:, :4]

def _write_kml_model(kml_path, name, lon, lat, altitude_m, dae_href="model.dae"):
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{_xml_escape(name)}</name>
    <Placemark>
      <name>{_xml_escape(name)}</name>
      <Model>
        <altitudeMode>{KML_ALTITUDE_MODE}</altitudeMode>
        <Location>
          <longitude>{float(lon):.15f}</longitude>
          <latitude>{float(lat):.15f}</latitude>
          <altitude>{float(altitude_m):.3f}</altitude>
        </Location>
        <Orientation>
          <heading>0.0</heading>
          <tilt>0.0</tilt>
          <roll>0.0</roll>
        </Orientation>
        <Scale>
          <x>1.0</x><y>1.0</y><z>1.0</z>
        </Scale>
        <Link>
          <href>{_xml_escape(dae_href)}</href>
        </Link>
      </Model>
    </Placemark>
  </Document>
</kml>
"""
    with open(kml_path, "w", encoding="utf-8") as f:
        f.write(kml)

def _write_textured_collada_scene(dae_path, meshes_named, name, texture_dir):
    effects_xml = []
    images_xml = []
    materials_xml = []
    geometries_xml = []
    nodes_xml = []
    texture_dir = Path(texture_dir)
    texture_dir.mkdir(parents=True, exist_ok=True)

    for mesh_idx, (mesh_name, mesh) in enumerate(meshes_named):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            continue

        base_id = _safe_collada_id(f"mesh_{mesh_idx}_{mesh_name}")
        geom_id = f"{base_id}_geom"
        positions_id = f"{base_id}_positions"
        vertices_id = f"{base_id}_vertices"
        mesh_xml_parts = [f"""
    <geometry id="{geom_id}" name="{_xml_escape(mesh_name)}">
      <mesh>
        <source id="{positions_id}">
          <float_array id="{positions_id}_array" count="{vertices.shape[0] * 3}">{_collada_float_list(vertices)}</float_array>
          <technique_common>
            <accessor source="#{positions_id}_array" count="{vertices.shape[0]}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="{vertices_id}">
          <input semantic="POSITION" source="#{positions_id}"/>
        </vertices>"""]

        tex_img = _mesh_texture_image(mesh)
        uv = _mesh_uv_array(mesh)
        if tex_img is not None and uv is not None:
            img_rel = f"textures/texture_{mesh_idx:04d}.png"
            # The mesh geometry defines the structural silhouette.  Strip the
            # generation-only alpha channel so KMZ viewers cannot infer an
            # undeclared cutout and punch facade/roof edge holes.
            tex_img.convert("RGB").save(
                texture_dir / f"texture_{mesh_idx:04d}.png"
            )
            image_id = f"{base_id}_image"
            effect_id = f"{base_id}_effect"
            material_id = f"{base_id}_material"
            surface_id = f"{base_id}_surface"
            sampler_id = f"{base_id}_sampler"
            texcoord_id = f"{base_id}_texcoords"
            images_xml.append(f"""
    <image id="{image_id}" name="{image_id}">
      <init_from>{_xml_escape(img_rel)}</init_from>
    </image>""")
            effects_xml.append(f"""
    <effect id="{effect_id}">
      <profile_COMMON>
        <newparam sid="{surface_id}">
          <surface type="2D">
            <init_from>{image_id}</init_from>
          </surface>
        </newparam>
        <newparam sid="{sampler_id}">
          <sampler2D>
            <source>{surface_id}</source>
          </sampler2D>
        </newparam>
        <technique sid="common">
          <phong>
            <ambient><color>0.350000 0.350000 0.350000 1.000000</color></ambient>
            <diffuse><texture texture="{sampler_id}" texcoord="UVSET0"/></diffuse>
            <specular><color>0.000000 0.000000 0.000000 1.000000</color></specular>
            <shininess><float>0.000000</float></shininess>
          </phong>
        </technique>
      </profile_COMMON>
      <extra>
        <technique profile="GOOGLEEARTH">
          <double_sided>1</double_sided>
        </technique>
      </extra>
    </effect>""")
            materials_xml.append(f"""
    <material id="{material_id}" name="{material_id}">
      <instance_effect url="#{effect_id}"/>
    </material>""")
            mesh_xml_parts.append(f"""
        <source id="{texcoord_id}">
          <float_array id="{texcoord_id}_array" count="{uv.shape[0] * 2}">{_collada_float_list(uv)}</float_array>
          <technique_common>
            <accessor source="#{texcoord_id}_array" count="{uv.shape[0]}" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>""")
            p_values = []
            for face in faces:
                for idx in face:
                    p_values.extend([int(idx), int(idx)])
            mesh_xml_parts.append(f"""
        <triangles material="{material_id}" count="{faces.shape[0]}">
          <input semantic="VERTEX" source="#{vertices_id}" offset="0"/>
          <input semantic="TEXCOORD" source="#{texcoord_id}" offset="1" set="0"/>
          <p>{" ".join(str(v) for v in p_values)}</p>
        </triangles>""")
            bind_material_xml = f"""
            <instance_material symbol="{material_id}" target="#{material_id}">
              <bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/>
            </instance_material>"""
        else:
            bind_materials = []
            colors = _mesh_face_colors(mesh)
            color_to_faces = defaultdict(list)
            for face_idx, rgba in enumerate(colors):
                color_to_faces[tuple(int(v) for v in rgba)].append(faces[face_idx])
            for color_idx, (rgba, color_faces) in enumerate(color_to_faces.items()):
                material_id = f"{base_id}_mat_{color_idx}"
                effect_id = f"{material_id}_effect"
                r, g, b, a = [v / 255.0 for v in rgba]
                ambient = [0.35 * r, 0.35 * g, 0.35 * b, a]
                effects_xml.append(f"""
    <effect id="{effect_id}">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <ambient><color>{_collada_float_list(ambient)}</color></ambient>
            <diffuse><color>{_collada_float_list([r, g, b, a])}</color></diffuse>
            <specular><color>0.000000 0.000000 0.000000 1.000000</color></specular>
            <shininess><float>0.000000</float></shininess>
          </phong>
        </technique>
      </profile_COMMON>
      <extra>
        <technique profile="GOOGLEEARTH">
          <double_sided>1</double_sided>
        </technique>
      </extra>
    </effect>""")
                materials_xml.append(f"""
    <material id="{material_id}" name="{material_id}">
      <instance_effect url="#{effect_id}"/>
    </material>""")
                color_faces = np.asarray(color_faces, dtype=np.int64)
                mesh_xml_parts.append(f"""
        <triangles material="{material_id}" count="{color_faces.shape[0]}">
          <input semantic="VERTEX" source="#{vertices_id}" offset="0"/>
          <p>{" ".join(str(int(v)) for v in color_faces.reshape(-1))}</p>
        </triangles>""")
                bind_materials.append(f'            <instance_material symbol="{material_id}" target="#{material_id}"/>')
            bind_material_xml = "\n".join(bind_materials)

        mesh_xml_parts.append("""
      </mesh>
    </geometry>""")
        geometries_xml.append("".join(mesh_xml_parts))
        nodes_xml.append(f"""
      <node id="{base_id}_node" name="{_xml_escape(mesh_name)}">
        <instance_geometry url="#{geom_id}">
          <bind_material>
            <technique_common>
{bind_material_xml}
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>""")

    dae = f"""<?xml version="1.0" encoding="UTF-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor>
      <authoring_tool>lod2_texture_pipeline</authoring_tool>
    </contributor>
    <unit name="meter" meter="1"/>
    <up_axis>Y_UP</up_axis>
  </asset>
  <library_images>
{''.join(images_xml)}
  </library_images>
  <library_effects>
{''.join(effects_xml)}
  </library_effects>
  <library_materials>
{''.join(materials_xml)}
  </library_materials>
  <library_geometries>
{''.join(geometries_xml)}
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
{''.join(nodes_xml)}
    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>
"""
    with open(dae_path, "w", encoding="utf-8") as f:
        f.write(dae)

def _zip_directory(source_dir, zip_path):
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(source_dir):
            for fn in files:
                full = Path(root) / fn
                z.write(full, arcname=full.relative_to(source_dir).as_posix())

def _epsg25832_to_lon_lat_height(easting, northing, height):
    lon, lat, z = transformer.transform(float(easting), float(northing), float(height))
    return float(lon), float(lat), float(z)

def _save_textured_kmz(meshes_named, output_kmz_path, name):
    if not EXPORT_KMZ or not meshes_named:
        return None

    relative_to_ground = str(KML_ALTITUDE_MODE).lower() == "relativetoground"
    origin_epsg = _make_export_origin(meshes_named, relative_to_ground=relative_to_ground)
    if origin_epsg is None:
        return None
    kml_altitude = 0.0 if relative_to_ground else None
    lon, lat, source_height = _epsg25832_to_lon_lat_height(origin_epsg[0], origin_epsg[1], origin_epsg[2])
    if kml_altitude is None:
        kml_altitude = source_height

    export_meshes = _copy_meshes_for_local_y_up(meshes_named, origin_epsg)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{_safe_artifact_folder_part(name)}_kmz_"))
    try:
        dae_path = tmp_dir / "model.dae"
        kml_path = tmp_dir / "doc.kml"
        meta_path = tmp_dir / "geo_meta.json"
        texture_dir = tmp_dir / "textures"

        _write_textured_collada_scene(dae_path, export_meshes, name, texture_dir)
        _write_kml_model(kml_path, name, lon, lat, kml_altitude, dae_href="model.dae")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "crs": SOURCE_CRS,
                "coordinate_origin_epsg_25832": [float(v) for v in origin_epsg],
                "anchor_lon_lat_height": [float(lon), float(lat), float(kml_altitude)],
                "source_origin_height": float(source_height),
                "altitude_mode": KML_ALTITUDE_MODE,
                "axis_mapping": "dae_x=east,dae_y=up,dae_z=-north",
            }, f, ensure_ascii=False, indent=2)

        _zip_directory(tmp_dir, output_kmz_path)
        return Path(output_kmz_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def _dedupe_xyz_path(points, tol=1e-6):
    out = []
    for p in points:
        p = np.asarray(p, dtype=np.float64)
        if len(out) == 0 or np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) >= 2 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out = out[:-1]
    return out

def _wall_records_compatible(a, b, max_normal_deg=28.0, max_dir_deg=28.0):
    qa = a["wall_quad"]
    qb = b["wall_quad"]
    da = _unit_xy(qa[1] - qa[0])
    db = _unit_xy(qb[1] - qb[0])
    na = _unit_xy(a["normal"])
    nb = _unit_xy(b["normal"])

    if np.linalg.norm(da) < 1e-9 or np.linalg.norm(db) < 1e-9:
        return False

    min_dir_dot = math.cos(math.radians(max_dir_deg))
    if abs(float(np.dot(da, db))) < min_dir_dot:
        return False

    if np.linalg.norm(na) >= 1e-9 and np.linalg.norm(nb) >= 1e-9:
        min_normal_dot = math.cos(math.radians(max_normal_deg))
        if float(np.dot(na, nb)) < min_normal_dot:
            return False

    return True

def _wall_record_base_path_xy(records):
    valid = [
        r for r in records
        if r.get("wall_quad") is not None and np.isfinite(r["wall_quad"]).all()
    ]
    if not valid:
        return np.zeros((0, 2), dtype=np.float64)
    pts = [np.asarray(valid[0]["wall_quad"][0, :2], dtype=np.float64)]
    pts.extend(np.asarray(r["wall_quad"][1, :2], dtype=np.float64) for r in valid)
    return np.asarray(_dedupe_xyz_path(pts, tol=1e-5), dtype=np.float64)

def _fit_wall_records_line_xy(records):
    pts = _wall_record_base_path_xy(records)
    if pts.shape[0] < 2:
        return None

    segs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    run_len = float(np.sum(seg_lens))
    if run_len < 1e-6:
        return None

    center = np.average(
        0.5 * (pts[:-1] + pts[1:]),
        axis=0,
        weights=np.maximum(seg_lens, 1e-6),
    )
    centered = pts - center

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = _unit_xy(vh[0])
    except np.linalg.LinAlgError:
        direction = _unit_xy(pts[-1] - pts[0])

    if np.linalg.norm(direction) < 1e-9:
        longest_idx = int(np.argmax(seg_lens))
        direction = _unit_xy(segs[longest_idx])
    if np.linalg.norm(direction) < 1e-9:
        return None

    if float(np.dot(direction, pts[-1] - pts[0])) < 0.0:
        direction = -direction

    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    distances = centered @ normal
    abs_distances = np.abs(distances)
    rms = float(np.sqrt(np.mean(abs_distances ** 2))) if abs_distances.size else 0.0
    along = centered @ direction

    return {
        "center": center,
        "direction": direction,
        "normal": normal,
        "max_deviation": float(np.max(abs_distances)) if abs_distances.size else 0.0,
        "rms_deviation": rms,
        "length": run_len,
        "span": float(np.max(along) - np.min(along)) if along.size else 0.0,
    }

def _wall_records_fit_facade_line(records):
    if not records:
        return False
    if len(records) == 1:
        q = records[0].get("wall_quad")
        return q is not None and np.isfinite(q).all()

    fit = _fit_wall_records_line_xy(records)
    if fit is None:
        return False

    max_dev = max(
        float(FACADE_GROUP_MAX_LINE_DEVIATION_M),
        min(2.0, fit["span"] * 0.035),
    )
    max_rms = max(
        float(FACADE_GROUP_MAX_LINE_RMS_M),
        min(0.9, fit["span"] * 0.018),
    )
    if fit["max_deviation"] > max_dev or fit["rms_deviation"] > max_rms:
        return False

    line_dir = fit["direction"]
    min_dir_dot = math.cos(math.radians(float(FACADE_GROUP_MAX_SEGMENT_ANGLE_DEG)))
    min_len = max(
        float(FACADE_GROUP_MIN_SEGMENT_LENGTH_M),
        min(2.0, fit["span"] * 0.08),
    )
    for rec in records:
        q = rec["wall_quad"]
        seg = q[1, :2] - q[0, :2]
        seg_len = float(np.linalg.norm(seg))
        if seg_len < min_len:
            continue
        seg_dir = _unit_xy(seg)
        if abs(float(np.dot(seg_dir, line_dir))) < min_dir_dot:
            return False

    normals = []
    for rec in records:
        n = _unit_xy(rec.get("normal", np.zeros(3))[:2])
        if np.linalg.norm(n) >= 1e-9:
            normals.append(n)
    if len(normals) >= 2:
        mean_n = _unit_xy(np.mean(np.vstack(normals), axis=0))
        if np.linalg.norm(mean_n) >= 1e-9:
            min_n_dot = math.cos(math.radians(float(FACADE_GROUP_MAX_NORMAL_ANGLE_DEG)))
            for n in normals:
                if float(np.dot(n, mean_n)) < min_n_dot:
                    return False

    return True

def _merge_line_compatible_groups(groups):
    groups = [list(g) for g in groups if g]
    if len(groups) <= 1:
        return groups

    changed = True
    while changed and len(groups) > 1:
        changed = False
        merged = []
        i = 0
        while i < len(groups):
            if i + 1 < len(groups) and _wall_records_fit_facade_line(groups[i] + groups[i + 1]):
                merged.append(groups[i] + groups[i + 1])
                i += 2
                changed = True
            else:
                merged.append(groups[i])
                i += 1
        groups = merged

        if len(groups) > 1 and _wall_records_fit_facade_line(groups[-1] + groups[0]):
            groups[0] = groups[-1] + groups[0]
            groups.pop()
            changed = True

    return groups

def _build_facade_groups(records):
    valid = [
        r for r in records
        if r.get("mesh_name") is not None
        and r.get("wall_quad") is not None
        and np.isfinite(r["wall_quad"]).all()
    ]
    if not valid:
        return []

    groups = []
    for rec in valid:
        if groups and _wall_records_fit_facade_line(groups[-1] + [rec]):
            groups[-1].append(rec)
        else:
            groups.append([rec])

    return _merge_line_compatible_groups(groups)

_DEBUG_GROUP_COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
    (58, 134, 255),
    (221, 85, 140),
    (82, 183, 136),
    (251, 86, 7),
    (106, 76, 147),
    (0, 150, 199),
]

def _debug_group_color(group_id):
    return _DEBUG_GROUP_COLORS[int(group_id) % len(_DEBUG_GROUP_COLORS)]

def _artifact_topology_id(value):
    """Return the one stable ID representation used by artifact filenames."""
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1

def _collect_facade_group_items(wall_records_by_loop):
    items = []
    group_id = 0
    for loop_key, records in wall_records_by_loop.items():
        for group_records in _build_facade_groups(records):
            items.append({
                "group_id": int(group_id),
                "loop_key": loop_key,
                "records": group_records,
            })
            group_id += 1
    return items

def _facade_group_artifact_debug_rows(facade_group_items, geojson_base):
    rows = []
    for item in facade_group_items:
        group_records = item.get("records", [])
        if not group_records:
            continue
        cid, lid = item.get("loop_key", (-1, -1))
        group_id = int(item.get("group_id", -1))
        wall_indices = [int(r["global_index"]) for r in group_records]
        cid_tag = _artifact_topology_id(cid)
        lid_tag = _artifact_topology_id(lid)
        facade_tag = f"c{cid_tag}_l{lid_tag}_g{group_id:02d}_w{wall_indices[0]:02d}-{wall_indices[-1]:02d}"
        for rec in group_records:
            gi = int(rec["global_index"])
            q = np.asarray(rec.get("wall_quad"), dtype=np.float64)
            row = {
                "geojson": geojson_base,
                "wall_tag": f"c{cid_tag}_l{lid_tag}_w{gi:02d}",
                "facade_group_tag": facade_tag,
                "facade_group_id": group_id,
                "facade_group_wall_indices": wall_indices,
                "facade_group_fragment_count": len(wall_indices),
                "component_id": int(rec.get("component_id", cid_tag)) if rec.get("component_id", cid_tag) is not None else -1,
                "loop_id": int(rec.get("loop_id", lid_tag)) if rec.get("loop_id", lid_tag) is not None else -1,
                "loop_index": int(rec.get("loop_index", -1)),
                "global_index": gi,
                "debug_only": True,
                "debug_status": "geometry_group_exists_no_texture_artifacts_yet",
            }
            if q.shape == (4, 3) and np.isfinite(q).all():
                row["wall_quad_xyz_b1b2t2t1"] = [[float(a), float(b), float(c)] for a, b, c in q.tolist()]
            rows.append(row)
    return rows

def _draw_dashed_line(draw, p0, p1, fill, width=2, dash=12, gap=7):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        draw.line(
            [(x0 + ux * t, y0 + uy * t), (x0 + ux * t2, y0 + uy * t2)],
            fill=fill,
            width=width,
        )
        t += dash + gap

def _draw_arrow(draw, p0, p1, fill, width=2):
    draw.line([p0, p1], fill=fill, width=width)
    x0, y0 = p0
    x1, y1 = p1
    angle = math.atan2(y1 - y0, x1 - x0)
    head = 10.0
    left = (
        x1 - head * math.cos(angle - math.pi / 6.0),
        y1 - head * math.sin(angle - math.pi / 6.0),
    )
    right = (
        x1 - head * math.cos(angle + math.pi / 6.0),
        y1 - head * math.sin(angle + math.pi / 6.0),
    )
    draw.polygon([p1, left, right], fill=fill)

def _draw_text_box(draw, xy, text, fill=(20, 20, 20, 255), bg=(255, 255, 255, 210)):
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
    except AttributeError:
        tw, th = draw.textsize(text)
        bbox = (x, y, x + tw, y + th)
    pad = 2
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=bg,
    )
    draw.text((x, y), text, fill=fill)

def _save_facade_group_debug_images(facade_group_items, per_building_out, geojson_base):
    if not SAVE_FACADE_GROUP_DEBUG_PNG or not facade_group_items:
        return

    records = []
    for item in facade_group_items:
        records.extend(item["records"])
    records = [
        r for r in records
        if r.get("wall_quad") is not None and np.isfinite(r["wall_quad"]).all()
    ]
    if not records:
        return

    all_xy = np.vstack([r["wall_quad"][:, :2] for r in records])
    xy_min = np.nanmin(all_xy, axis=0)
    xy_max = np.nanmax(all_xy, axis=0)
    span = xy_max - xy_min
    max_span = max(float(np.nanmax(span)), 1.0)
    pad_m = max(max_span * 0.08, 0.5)
    xy_min -= pad_m
    xy_max += pad_m
    span = xy_max - xy_min
    span[span < 1e-6] = 1.0

    plot_w = 1260
    plot_h = 1000
    legend_w = 560
    margin = 60
    canvas = Image.new("RGBA", (plot_w + legend_w, plot_h), (250, 250, 248, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    scale = min(
        (plot_w - 2 * margin) / float(span[0]),
        (plot_h - 2 * margin) / float(span[1]),
    )

    def xy_to_px(xy):
        x = margin + (float(xy[0]) - xy_min[0]) * scale
        y = plot_h - margin - (float(xy[1]) - xy_min[1]) * scale
        return (x, y)

    draw.rectangle([margin, margin, plot_w - margin, plot_h - margin], outline=(210, 210, 210, 255), width=1)
    draw.text((24, 18), f"{geojson_base} - facade grouping top-down debug", fill=(20, 20, 20, 255))
    draw.text(
        (24, 38),
        "thick solid = fragment edge, thin centered = fitted facade side, dashed offset = roof/top edge, arrow = outward normal",
        fill=(70, 70, 70, 255),
    )

    # Background: every valid fragment in light gray.
    for rec in records:
        q = rec["wall_quad"]
        draw.line([xy_to_px(q[0, :2]), xy_to_px(q[1, :2])], fill=(190, 190, 190, 180), width=12)

    top_offset_m = max(max_span * 0.012, 0.25)
    arrow_len_m = max(max_span * 0.045, 1.0)

    metadata = []
    for item in facade_group_items:
        gid = int(item["group_id"])
        color = _debug_group_color(gid)
        color_rgba = (*color, 235)
        text_color = (15, 15, 15, 255)
        group_records = [
            r for r in item["records"]
            if r.get("wall_quad") is not None and np.isfinite(r["wall_quad"]).all()
        ]
        if not group_records:
            continue
        line_fit = _fit_wall_records_line_xy(group_records)
        fit_line_xy = None
        if line_fit is not None:
            group_path_xy = _wall_record_base_path_xy(group_records)
            along = (group_path_xy - line_fit["center"]) @ line_fit["direction"]
            fit_line_xy = np.vstack([
                line_fit["center"] + line_fit["direction"] * float(np.nanmin(along)),
                line_fit["center"] + line_fit["direction"] * float(np.nanmax(along)),
            ])

        for rec in group_records:
            q = rec["wall_quad"]
            n_xy = _unit_xy(rec.get("normal", np.zeros(3))[:2])
            center_xy = 0.5 * (q[0, :2] + q[1, :2])

            draw.line(
                [xy_to_px(q[0, :2]), xy_to_px(q[1, :2])],
                fill=(255, 255, 255, 255),
                width=12,
            )
            draw.line(
                [xy_to_px(q[0, :2]), xy_to_px(q[1, :2])],
                fill=color_rgba,
                width=8,
            )

            if np.linalg.norm(n_xy) >= 1e-9:
                roof_a = q[3, :2] + n_xy * top_offset_m
                roof_b = q[2, :2] + n_xy * top_offset_m
                _draw_dashed_line(
                    draw,
                    xy_to_px(roof_a),
                    xy_to_px(roof_b),
                    fill=(*color, 255),
                    width=3,
                )
                _draw_arrow(
                    draw,
                    xy_to_px(center_xy),
                    xy_to_px(center_xy + n_xy * arrow_len_m),
                    fill=(45, 45, 45, 210),
                    width=2,
                )

            label_xy = center_xy + (n_xy * top_offset_m * 2.0 if np.linalg.norm(n_xy) >= 1e-9 else 0.0)
            _draw_text_box(
                draw,
                xy_to_px(label_xy),
                f"G{gid} w{int(rec['global_index'])}",
                fill=text_color,
            )

        if fit_line_xy is not None:
            draw.line(
                [xy_to_px(fit_line_xy[0]), xy_to_px(fit_line_xy[1])],
                fill=(255, 255, 255, 255),
                width=5,
            )
            draw.line(
                [xy_to_px(fit_line_xy[0]), xy_to_px(fit_line_xy[1])],
                fill=(*color, 255),
                width=3,
            )

        cid, lid = item["loop_key"]
        metadata.append({
            "group_id": gid,
            "component_id": int(cid) if cid is not None else -1,
            "loop_id": int(lid) if lid is not None else -1,
            "wall_global_indices": [int(r["global_index"]) for r in group_records],
            "wall_loop_indices": [int(r["loop_index"]) for r in group_records],
            "wall_edges_source_target": [
                [int(r["edge"][0]), int(r["edge"][1])] for r in group_records
            ],
            "next_edges_source_target": [
                [int(r["next_edge"][0]), int(r["next_edge"][1])] for r in group_records
            ],
            "normal_xy_mean": [
                float(v) for v in _unit_xy(np.nanmean(np.vstack([r["normal"][:2] for r in group_records]), axis=0)).tolist()
            ],
            "line_fit": None if line_fit is None else {
                "center_xy": [float(v) for v in line_fit["center"].tolist()],
                "direction_xy": [float(v) for v in line_fit["direction"].tolist()],
                "max_deviation_m": float(line_fit["max_deviation"]),
                "rms_deviation_m": float(line_fit["rms_deviation"]),
                "span_m": float(line_fit["span"]),
                "length_m": float(line_fit["length"]),
            },
            "fragment_count": int(len(group_records)),
        })

    legend_x = plot_w + 24
    y = 24
    draw.text((legend_x, y), "Facade groups", fill=(20, 20, 20, 255))
    y += 28
    for item in facade_group_items[:40]:
        gid = int(item["group_id"])
        group_records = item["records"]
        color = _debug_group_color(gid)
        wall_ids = ",".join(str(int(r["global_index"])) for r in group_records)
        cid, lid = item["loop_key"]
        draw.rectangle([legend_x, y + 3, legend_x + 18, y + 21], fill=(*color, 255), outline=(0, 0, 0, 255))
        draw.text(
            (legend_x + 28, y),
            f"G{gid} c{cid} l{lid}: walls {wall_ids}",
            fill=(30, 30, 30, 255),
        )
        y += 24
        if y > plot_h - 30:
            draw.text((legend_x, y), "... legend truncated", fill=(80, 80, 80, 255))
            break

    topdown_path = Path(per_building_out) / f"{geojson_base}__debug_facade_groups_topdown.png"
    canvas.convert("RGB").save(topdown_path)

    _save_facade_group_unwrapped_debug(facade_group_items, per_building_out, geojson_base)

    meta_path = Path(per_building_out) / f"{geojson_base}__debug_facade_groups.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved facade group debug: {topdown_path.name}")

def _save_facade_group_unwrapped_debug(facade_group_items, per_building_out, geojson_base):
    valid_items = []
    for item in facade_group_items:
        geom = _facade_group_geometry(item["records"])
        if geom is not None:
            valid_items.append((item, geom))
    if not valid_items:
        return

    width = 1700
    row_h = 220
    header_h = 70
    height = header_h + row_h * len(valid_items) + 30
    canvas = Image.new("RGBA", (width, height), (250, 250, 248, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((24, 20), f"{geojson_base} - facade grouping unwrapped debug", fill=(20, 20, 20, 255))
    draw.text((24, 42), "Each row is one grouped facade plane. Colored polygons are wall fragments in group metric coordinates.", fill=(70, 70, 70, 255))

    label_w = 250
    panel_x = label_w + 24
    panel_w = width - panel_x - 40
    panel_h = row_h - 54

    for row_idx, (item, geom) in enumerate(valid_items):
        y0 = header_h + row_idx * row_h
        gid = int(item["group_id"])
        color = _debug_group_color(gid)
        group_records = item["records"]
        frame = geom["frame"]

        pts = []
        for rec in group_records:
            if rec.get("wall_quad") is not None and np.isfinite(rec["wall_quad"]).all():
                pts.append(frame["to_uv"](rec["wall_quad"]))
        if not pts:
            continue
        pts_m = np.vstack(pts)
        uv_min = np.nanmin(pts_m, axis=0)
        uv_max = np.nanmax(pts_m, axis=0)
        span = uv_max - uv_min
        span[span < 1e-6] = 1.0
        pad_u = max(float(span[0]) * 0.04, 0.1)
        pad_v = max(float(span[1]) * 0.10, 0.1)
        uv_min -= np.array([pad_u, pad_v], dtype=np.float64)
        uv_max += np.array([pad_u, pad_v], dtype=np.float64)
        span = uv_max - uv_min

        panel_y = y0 + 36
        scale = min(panel_w / float(span[0]), panel_h / float(span[1]))

        def uv_to_px(uv):
            x = panel_x + (float(uv[0]) - uv_min[0]) * scale
            y = panel_y + panel_h - (float(uv[1]) - uv_min[1]) * scale
            return (x, y)

        draw.rectangle([20, y0 + 8, width - 20, y0 + row_h - 8], outline=(220, 220, 220, 255), width=1)
        cid, lid = item["loop_key"]
        wall_ids = ",".join(str(int(r["global_index"])) for r in group_records)
        draw.rectangle([24, y0 + 22, 44, y0 + 42], fill=(*color, 255), outline=(0, 0, 0, 255))
        draw.text((54, y0 + 18), f"G{gid} c{cid} l{lid}", fill=(20, 20, 20, 255))
        draw.text((54, y0 + 40), f"walls {wall_ids}", fill=(70, 70, 70, 255))
        draw.text(
            (54, y0 + 62),
            f"width {span[0]:.2f}m  height {span[1]:.2f}m",
            fill=(70, 70, 70, 255),
        )
        draw.rectangle([panel_x, panel_y, panel_x + panel_w, panel_y + panel_h], outline=(210, 210, 210, 255), width=1)

        outline_px = [uv_to_px(p) for p in geom["outline_m"]]
        if len(outline_px) >= 3:
            draw.polygon(outline_px, fill=(235, 235, 235, 180), outline=(60, 60, 60, 255))

        for rec in group_records:
            q = rec.get("wall_quad")
            if q is None or not np.isfinite(q).all():
                continue
            poly_m = frame["to_uv"](q)
            poly_px = [uv_to_px(p) for p in poly_m]
            draw.polygon(poly_px, fill=(*color, 72), outline=(*color, 255))
            center_m = np.nanmean(poly_m, axis=0)
            _draw_text_box(
                draw,
                uv_to_px(center_m),
                f"w{int(rec['global_index'])}",
                fill=(15, 15, 15, 255),
                bg=(255, 255, 255, 190),
            )

    unwrapped_path = Path(per_building_out) / f"{geojson_base}__debug_facade_groups_unwrapped.png"
    canvas.convert("RGB").save(unwrapped_path)
    print(f"Saved facade group debug: {unwrapped_path.name}")

def _facade_outline_xyz(group_records):
    bottom = [group_records[0]["wall_quad"][0]]
    bottom.extend(rec["wall_quad"][1] for rec in group_records)
    top = [rec["wall_quad"][2] for rec in reversed(group_records)]
    top.append(group_records[0]["wall_quad"][3])
    return np.asarray(_dedupe_xyz_path(bottom + top), dtype=np.float64)

def _facade_frame(group_records):
    base_pts = []
    normals = []
    for rec in group_records:
        q = rec["wall_quad"]
        base_pts.extend([q[0], q[1]])
        nxy = _unit_xy(rec["normal"])
        if np.linalg.norm(nxy) >= 1e-9:
            normals.append(nxy)

    base_arr = np.asarray(base_pts, dtype=np.float64)
    first_base = group_records[0]["wall_quad"][0]
    last_base = group_records[-1]["wall_quad"][1]
    line_fit = _fit_wall_records_line_xy(group_records)
    u_xy = _unit_xy(line_fit["direction"]) if line_fit is not None else _unit_xy(last_base[:2] - first_base[:2])

    if np.linalg.norm(u_xy) < 1e-9 and len(base_arr) >= 2:
        xy = base_arr[:, :2]
        centered = xy - xy.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            u_xy = _unit_xy(vh[0])
            if np.dot(u_xy, last_base[:2] - first_base[:2]) < 0:
                u_xy = -u_xy
        except np.linalg.LinAlgError:
            u_xy = np.array([1.0, 0.0], dtype=float)

    if np.linalg.norm(u_xy) < 1e-9:
        u_xy = np.array([1.0, 0.0], dtype=float)

    if normals:
        n_xy = _unit_xy(np.mean(np.vstack(normals), axis=0))
    else:
        n_xy = np.array([-u_xy[1], u_xy[0]], dtype=float)

    if np.linalg.norm(n_xy) < 1e-9:
        n_xy = np.array([-u_xy[1], u_xy[0]], dtype=float)

    if line_fit is not None:
        along = (base_arr[:, :2] - line_fit["center"]) @ u_xy
        origin_xy = line_fit["center"] + u_xy * float(np.nanmin(along))
    else:
        origin_xy = first_base[:2]

    origin = np.array([
        float(origin_xy[0]),
        float(origin_xy[1]),
        float(np.nanmin(base_arr[:, 2])),
    ], dtype=np.float64)
    u_dir = np.array([float(u_xy[0]), float(u_xy[1]), 0.0], dtype=np.float64)
    up_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def to_uv(points_xyz):
        pts = np.asarray(points_xyz, dtype=np.float64)
        single = pts.ndim == 1
        pts2 = pts.reshape(-1, 3)
        d = pts2 - origin
        uv = np.column_stack([d @ u_dir, d @ up_dir])
        return uv[0] if single else uv

    def to_xyz(points_uv):
        uv = np.asarray(points_uv, dtype=np.float64)
        single = uv.ndim == 1
        uv2 = uv.reshape(-1, 2)
        xyz = origin + uv2[:, 0:1] * u_dir + uv2[:, 1:2] * up_dir
        return xyz[0] if single else xyz

    return {
        "origin": origin,
        "u_dir": u_dir,
        "up_dir": up_dir,
        "normal_xy": n_xy,
        "to_uv": to_uv,
        "to_xyz": to_xyz,
    }

def _facade_group_geometry(group_records):
    outline_xyz = _facade_outline_xyz(group_records)
    if outline_xyz.shape[0] < 3:
        return None

    frame = _facade_frame(group_records)
    outline_m = frame["to_uv"](outline_xyz)

    all_pts = []
    for rec in group_records:
        all_pts.extend(rec["wall_quad"])
    all_m = frame["to_uv"](np.asarray(all_pts, dtype=np.float64))

    u_min, v_min = np.nanmin(all_m, axis=0)
    u_max, v_max = np.nanmax(all_m, axis=0)
    if (u_max - u_min) < 1e-6 or (v_max - v_min) < 1e-6:
        return None

    rect_m = np.array([
        [u_min, v_min],
        [u_max, v_min],
        [u_max, v_max],
        [u_min, v_max],
    ], dtype=np.float64)
    rect_xyz = frame["to_xyz"](rect_m)

    return {
        "frame": frame,
        "outline_xyz": outline_xyz,
        "outline_m": outline_m,
        "rect_m": rect_m,
        "rect_xyz": rect_xyz,
    }

def _extract_mask_stack(out_obj, H, W):
    masks = None
    if isinstance(out_obj, dict):
        masks = out_obj.get("masks", out_obj.get("mask", out_obj.get("pred_masks", None)))
    else:
        masks = getattr(out_obj, "masks", None)

    if masks is None:
        return np.zeros((0, H, W), dtype=bool)

    if torch.is_tensor(masks):
        m = masks.detach().float().cpu().numpy()
    else:
        m = np.asarray(masks)

    if m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0]
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim != 3:
        return np.zeros((0, H, W), dtype=bool)

    stack = (m > 0.5)
    keep = [stack[i] for i in range(stack.shape[0]) if stack[i].any()]
    if len(keep) == 0:
        return np.zeros((0, H, W), dtype=bool)
    return np.stack(keep, axis=0)

def _stack_union(mask_stack, H, W):
    if mask_stack.shape[0] == 0:
        return np.zeros((H, W), dtype=bool)
    return mask_stack.any(axis=0)

def _polygon_to_mask(H, W, poly_xy):
    mask = np.zeros((H, W), dtype=np.uint8)
    poly = np.round(poly_xy).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255)
    return mask > 0

def _remove_external_building_pixels(base_mask, external_mask):
    """Return retained pixels and the external obstruction removed inside them."""
    base = np.asarray(base_mask, dtype=bool)
    if external_mask is None:
        return base.copy(), np.zeros_like(base, dtype=bool)
    external = np.asarray(external_mask, dtype=bool)
    if external.shape != base.shape:
        raise ValueError("External-building mask must match the selected source mask.")
    removed = base & external
    return base & ~removed, removed


def _external_building_lr_side_exclusion_mask(external_mask, target_mask):
    """
    Extend an OSM obstruction as an LR-style side crop.

    The OSM mask initially contains only pixels where the external building
    overlaps the corrected target projection. Fit the shared obstruction /
    visible-target divider, extend that line to the image top and bottom, and
    exclude the half-plane on the obstruction side. This preserves image
    coordinates without turning the obstruction's x-range into a boxy strip.
    """
    obstruction = np.asarray(external_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if obstruction.ndim != 2 or target.ndim != 2:
        raise ValueError("External-building side exclusion expects two 2D masks.")
    if obstruction.shape != target.shape:
        raise ValueError("External-building and target masks must have the same shape.")

    obstruction = obstruction & target
    visible_target = target & ~obstruction
    info = {
        "mode": "lr_style_obstruction_side_crop",
        "applied": False,
        "reason": "empty_obstruction",
        "divider_lines": [],
        "raw_occlusion_pixel_count": int(obstruction.sum()),
        "visible_target_pixel_count": int(visible_target.sum()),
    }
    if not obstruction.any():
        return np.zeros_like(obstruction), info
    if not visible_target.any():
        info["reason"] = "no_visible_target_for_obstruction_divider"
        return obstruction.copy(), info

    visible_neighborhood = cv2.dilate(
        visible_target.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ) > 0
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        obstruction.astype(np.uint8),
        connectivity=8,
    )
    height, width = obstruction.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    side_exclusion = np.zeros_like(obstruction)
    divider_lines = []

    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 3:
            continue

        frontier = component & visible_neighborhood
        frontier_y, frontier_x = np.nonzero(frontier)
        if len(frontier_x) < 4:
            continue

        points = np.column_stack([frontier_x, frontier_y]).astype(np.float32)
        vx, vy, x0, y0 = (
            float(value)
            for value in cv2.fitLine(
                points,
                cv2.DIST_HUBER,
                0.0,
                0.01,
                0.01,
            ).reshape(-1)
        )
        norm = math.hypot(vx, vy)
        if norm <= 1e-9:
            continue
        vx /= norm
        vy /= norm

        # LR crops have a divider that runs from the image top to bottom.
        # A near-horizontal frontier cannot define that operation reliably.
        if abs(vy) < 0.15:
            continue

        signed_distance = (-vy * (xx - x0)) + (vx * (yy - y0))
        candidate_sides = (
            ("positive", signed_distance >= 0.0),
            ("negative", signed_distance <= 0.0),
        )
        best = None
        for side_name, candidate_side in candidate_sides:
            obstruction_capture = float(
                np.count_nonzero(candidate_side & component) / component_area
            )
            visible_loss = float(
                np.count_nonzero(candidate_side & visible_target)
                / max(int(visible_target.sum()), 1)
            )
            score = obstruction_capture - 2.0 * visible_loss
            row = (score, obstruction_capture, -visible_loss, side_name, candidate_side)
            if best is None or row[:4] > best[:4]:
                best = row

        if best is None:
            continue
        _score, obstruction_capture, negative_visible_loss, side_name, crop_side = best
        visible_loss = -float(negative_visible_loss)
        if obstruction_capture < 0.75 or visible_loss > 0.35:
            continue

        x_top = x0 + vx * ((0.0 - y0) / vy)
        x_bottom = x0 + vx * (((height - 1.0) - y0) / vy)
        side_exclusion |= crop_side
        divider_lines.append({
            "component_label": int(label),
            "component_area_px": component_area,
            "removed_side": side_name,
            "top_xy": [float(x_top), 0.0],
            "bottom_xy": [float(x_bottom), float(height - 1)],
            "obstruction_capture_fraction": float(obstruction_capture),
            "visible_target_removed_fraction": float(visible_loss),
        })

    if not divider_lines:
        info["reason"] = "no_reliable_lr_style_obstruction_divider"
        return obstruction.copy(), info

    # Preserve every positively identified obstruction pixel even when its
    # rasterized edge falls a pixel across the fitted divider.
    side_exclusion |= obstruction
    info.update({
        "applied": True,
        "reason": "obstruction_side_half_plane_extended_top_to_bottom",
        "divider_lines": divider_lines,
        "excluded_pixel_count": int(side_exclusion.sum()),
        "excluded_target_pixel_count": int((side_exclusion & target).sum()),
        "remaining_target_pixel_count": int((target & ~side_exclusion).sum()),
    })
    return side_exclusion, info


def _dilate_bool_mask(mask_bool, radius_px):
    radius = int(max(0, round(float(radius_px))))
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_bool.astype(np.uint8) * 255, kernel, iterations=1) > 0

def _apply_lr_alpha_gate_to_selected_mask(selected_mask, lr_alpha_u8, wall_poly_xy):
    """
    Keep the LR band as a SAM crop / candidate-selection aid, but do not let it
    erase pixels after SAM has selected the target facade mask. The selected
    mask is already filtered by wall-overlap scoring, so intersecting it again
    with the LR band can wrongly cut valid facade texture.
    """
    selected_mask = np.asarray(selected_mask, dtype=bool)
    H, W = selected_mask.shape[:2]
    lr_allowed = np.asarray(lr_alpha_u8, dtype=np.uint8) > 0
    if lr_allowed.shape != selected_mask.shape:
        lr_allowed = np.ones_like(selected_mask, dtype=bool)

    selected_px = int(selected_mask.sum())
    raw_lr_kept_px = int((selected_mask & lr_allowed).sum())

    if not bool(LR_BAND_PROTECT_SELECTED_SEGMENTATION):
        gated = selected_mask & lr_allowed
        return gated, {
            "enabled": False,
            "mode": "raw_lr_alpha_intersection",
            "selected_px_before_gate": selected_px,
            "kept_by_raw_lr_alpha_px": raw_lr_kept_px,
            "rescued_selected_px": 0,
            "removed_selected_px": int(selected_px - int(gated.sum())),
            "wall_margin_px": 0,
        }

    margin_px = int(max(0, round(float(LR_BAND_SELECTED_SEGMENT_MARGIN_PX))))
    rescued_px = int((selected_mask & (~lr_allowed)).sum())
    info = {
        "enabled": True,
        "mode": "preserve_selected_sam_mask",
        "selected_px_before_gate": selected_px,
        "kept_by_raw_lr_alpha_px": raw_lr_kept_px,
        "rescued_selected_px": rescued_px,
        "removed_selected_px": 0,
        "wall_margin_px": margin_px,
    }
    return selected_mask, info

def _lr_model_crop_bbox(alpha_u8, wall_poly_xy, width, height):
    alpha_bbox = Image.fromarray(np.asarray(alpha_u8, dtype=np.uint8)).getbbox()
    if alpha_bbox is None:
        alpha_bbox = (0, 0, int(width), int(height))

    poly = np.asarray(wall_poly_xy, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2 or not np.isfinite(poly).all():
        return alpha_bbox

    margin = int(max(0, round(float(LR_BAND_SELECTED_SEGMENT_MARGIN_PX))))
    x0 = int(np.floor(np.nanmin(poly[:, 0]))) - margin
    y0 = int(np.floor(np.nanmin(poly[:, 1]))) - margin
    x1 = int(np.ceil(np.nanmax(poly[:, 0]))) + margin
    y1 = int(np.ceil(np.nanmax(poly[:, 1]))) + margin

    L = max(0, min(int(alpha_bbox[0]), x0))
    T = max(0, min(int(alpha_bbox[1]), y0))
    R = min(int(width), max(int(alpha_bbox[2]), x1))
    B = min(int(height), max(int(alpha_bbox[3]), y1))

    if R <= L or B <= T:
        return alpha_bbox
    return (L, T, R, B)

def _sample_polyline_points(points, max_points=2500):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] <= max_points:
        return pts
    idx = np.linspace(0, pts.shape[0] - 1, int(max_points)).astype(np.int64)
    return pts[idx]

def _extract_contour_points_from_mask(mask_bool, max_points=2500):
    mask_u8 = (np.asarray(mask_bool, dtype=bool).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= QUAD_MIN_CONTOUR_AREA_PX]
    if not contours:
        return None, []
    pts = np.vstack([c[:, 0, :] for c in contours]).astype(np.float64)
    return _sample_polyline_points(pts, max_points=max_points), contours

def _min_signed_dist_points_to_polygon(pts, polygon_xy):
    contour = np.asarray(polygon_xy, dtype=np.float32).reshape((-1, 1, 2))
    dmin = 1e18
    for p in np.asarray(pts, dtype=np.float64):
        d = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)
        dmin = min(dmin, float(d))
    return float(dmin)

def _ortho_fit_enabled():
    return bool(globals().get("ENABLE_ORTHO_FIT", globals().get("ENABLE_ORTHO_POLYGON_FIT", True)))

def _fit_ortho_rgba_alpha_inside_polygon(
    ortho_rgba,
    wall_poly_px,
    source_mask_override=None,
    max_scale_delta=None,
    max_translation_px=None,
):
    """
    Fit the rectified content-mask contour inside the facade outline using a
    uniform scale + translation, then apply that affine to RGBA. A four-point
    wall is handled as the quad case of the same polygon fitting framework.
    """
    enabled = _ortho_fit_enabled()
    info = {
        "enabled": bool(enabled),
        "applied": False,
        "fit_mode": "uniform_scale_plus_translation_inside_polygon",
        "scale": 1.0,
        "tx": 0.0,
        "ty": 0.0,
        "min_signed_dist_px": None,
        "center_dist": None,
        "source_area_px": 0,
        "target_area_px": 0,
        "reason": None,
    }
    if not enabled:
        info["reason"] = "disabled"
        source_pts = None
        if ortho_rgba is not None and ortho_rgba.ndim == 3 and ortho_rgba.shape[2] == 4:
            height, width = ortho_rgba.shape[:2]
            wall_mask = build_wall_region_mask(height, width, wall_poly_px) > 0
            info["target_area_px"] = int(wall_mask.sum())
            if source_mask_override is not None:
                debug_mask = np.asarray(source_mask_override, dtype=bool)
                if debug_mask.shape == (height, width):
                    info["source_mask"] = (
                        "reused_prefit_semantic_mask_inside_projection"
                    )
                else:
                    debug_mask = ortho_rgba[:, :, 3] > 0
                    info["source_mask"] = "rgba_alpha_shape_fallback"
            else:
                debug_mask = ortho_rgba[:, :, 3] > 0
                info["source_mask"] = "rgba_alpha"
            info["source_area_px"] = int(debug_mask.sum())
            source_pts, _ = _extract_contour_points_from_mask(
                debug_mask,
                max_points=int(POLYGON_FIT_MAX_CONTOUR_POINTS),
            )
        return ortho_rgba, None, source_pts, None, info

    if ortho_rgba is None or ortho_rgba.ndim != 3 or ortho_rgba.shape[2] != 4:
        info["reason"] = "invalid_rgba"
        return ortho_rgba, None, None, None, info

    H, W = ortho_rgba.shape[:2]
    wall_mask = build_wall_region_mask(H, W, wall_poly_px) > 0
    target_area = int(wall_mask.sum())
    info["target_area_px"] = target_area
    if target_area <= 0:
        info["reason"] = "empty_target_polygon"
        return ortho_rgba, None, None, None, info

    if source_mask_override is None:
        alpha_mask_raw = ortho_rgba[:, :, 3] > 0
        info["source_mask"] = "rgba_alpha"
    else:
        alpha_mask_raw = np.asarray(source_mask_override, dtype=bool)
        info["source_mask"] = "reused_prefit_semantic_mask_inside_projection"
        if alpha_mask_raw.shape != (H, W):
            info["reason"] = "source_mask_shape_mismatch"
            return ortho_rgba, None, None, None, info
    if not alpha_mask_raw.any():
        info["reason"] = "empty_source_alpha"
        return ortho_rgba, None, None, None, info

    alpha_mask = clean_selected_mask(alpha_mask_raw)
    source_area = int(alpha_mask.sum())
    info["source_area_px"] = source_area
    if source_area <= 0:
        info["reason"] = "empty_source_after_clean"
        return ortho_rgba, None, None, None, info

    source_pts, source_contours = _extract_contour_points_from_mask(
        alpha_mask,
        max_points=int(POLYGON_FIT_MAX_CONTOUR_POINTS)
    )
    if source_pts is None or source_pts.shape[0] < 4:
        info["reason"] = "not_enough_source_contour_points"
        return ortho_rgba, None, None, None, info

    sy, sx = np.where(alpha_mask)
    ty, tx = np.where(wall_mask)
    source_center = np.array([float(sx.mean()), float(sy.mean())], dtype=np.float64)
    target_center = np.array([float(tx.mean()), float(ty.mean())], dtype=np.float64)
    source_centered = source_pts - source_center

    target_xy = np.column_stack([tx, ty]).astype(np.float64)
    tmin = target_xy.min(axis=0)
    tmax = target_xy.max(axis=0)
    target_w = max(float(tmax[0] - tmin[0]), 1.0)
    target_h = max(float(tmax[1] - tmin[1]), 1.0)
    target_diag = max(float(np.hypot(target_w, target_h)), 1.0)

    source_area_f = max(float(source_area), 1.0)
    target_area_f = max(float(target_area), 1.0)
    s_guess = math.sqrt(target_area_f / source_area_f)

    center_dx_vals = np.linspace(
        -PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * target_w,
        +PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * target_w,
        POLYGON_FIT_CENTER_SHIFT_STEPS,
    )
    center_dy_vals = np.linspace(
        -PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * target_h,
        +PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * target_h,
        POLYGON_FIT_CENTER_SHIFT_STEPS,
    )

    best_scale = -1.0
    best_center = target_center.copy()
    best_center_dist = 1e18
    best_min_dist = -1e18
    best_pts = source_pts.copy()

    def build_points(center_xy, scale):
        return source_centered * scale + center_xy

    def is_inside(points):
        dmin = _min_signed_dist_points_to_polygon(points, wall_poly_px)
        return dmin >= PERSPECTIVE_FIT_INSET_PX, dmin

    for dx in center_dx_vals:
        for dy in center_dy_vals:
            center_xy = target_center + np.array([dx, dy], dtype=np.float64)
            s_lo = 0.0
            s_hi = max(0.05, s_guess)

            pts_hi = build_points(center_xy, s_hi)
            ok_hi, _d_hi = is_inside(pts_hi)

            grow_iter = 0
            while ok_hi and s_hi < PERSPECTIVE_FIT_MAX_SCALE and grow_iter < 20:
                s_lo = s_hi
                s_hi *= PERSPECTIVE_FIT_SCALE_GROWTH
                pts_hi = build_points(center_xy, s_hi)
                ok_hi, _d_hi = is_inside(pts_hi)
                grow_iter += 1

            lo = s_lo
            hi = s_hi
            for _ in range(POLYGON_FIT_BINARY_STEPS):
                mid = 0.5 * (lo + hi)
                pts_mid = build_points(center_xy, mid)
                ok_mid, _d_mid = is_inside(pts_mid)
                if ok_mid:
                    lo = mid
                else:
                    hi = mid

            s_best_here = lo
            pts_best_here = build_points(center_xy, s_best_here)
            _, d_best_here = is_inside(pts_best_here)
            center_dist_here = float(np.linalg.norm(center_xy - target_center) / target_diag)

            if (
                (s_best_here > best_scale + 1e-9) or
                (abs(s_best_here - best_scale) <= 1e-9 and center_dist_here < best_center_dist - 1e-9) or
                (abs(s_best_here - best_scale) <= 1e-9 and abs(center_dist_here - best_center_dist) <= 1e-9 and d_best_here < best_min_dist)
            ):
                best_scale = float(s_best_here)
                best_center = center_xy.copy()
                best_center_dist = center_dist_here
                best_min_dist = float(d_best_here)
                best_pts = pts_best_here

    if best_scale <= 0.0:
        info["reason"] = "no_positive_scale_found"
        return ortho_rgba, None, source_pts, None, info

    tx_aff = float(best_center[0] - best_scale * source_center[0])
    ty_aff = float(best_center[1] - best_scale * source_center[1])
    M = np.array([
        [best_scale, 0.0, tx_aff],
        [0.0, best_scale, ty_aff],
    ], dtype=np.float64)

    scale_delta = abs(float(best_scale) - 1.0)
    translation_px = float(np.hypot(tx_aff, ty_aff))
    info.update({
        "scale_delta": scale_delta,
        "translation_px": translation_px,
        "max_scale_delta": (
            float(max_scale_delta) if max_scale_delta is not None else None
        ),
        "max_translation_px": (
            float(max_translation_px) if max_translation_px is not None else None
        ),
    })
    if max_scale_delta is not None and scale_delta > float(max_scale_delta):
        info["reason"] = "rejected_scale_delta_limit"
        return ortho_rgba, None, source_pts, best_pts, info
    if max_translation_px is not None and translation_px > float(max_translation_px):
        info["reason"] = "rejected_translation_limit"
        return ortho_rgba, None, source_pts, best_pts, info

    src_for_warp = ortho_rgba.copy()
    if source_mask_override is None:
        src_for_warp[:, :, 3] = (alpha_mask.astype(np.uint8) * 255)
    src_bgra = cv2.cvtColor(src_for_warp, cv2.COLOR_RGBA2BGRA)
    warped_bgra = cv2.warpAffine(
        src_bgra,
        M.astype(np.float32),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    warped_rgba = cv2.cvtColor(warped_bgra, cv2.COLOR_BGRA2RGBA)

    info.update({
        "applied": True,
        "scale": float(best_scale),
        "tx": tx_aff,
        "ty": ty_aff,
        "final_center_x": float(best_center[0]),
        "final_center_y": float(best_center[1]),
        "min_signed_dist_px": float(best_min_dist),
        "center_dist": float(best_center_dist),
        "source_center_x": float(source_center[0]),
        "source_center_y": float(source_center[1]),
        "target_center_x": float(target_center[0]),
        "target_center_y": float(target_center[1]),
        "reason": "applied",
    })
    return warped_rgba, M, source_pts, best_pts, info

def _save_ortho_fit_debug_overlay(
    img_rgba,
    wall_poly_px,
    source_pts,
    fitted_pts,
    out_path,
    fit_info=None,
    source_mask=None,
    display_mask=None,
):
    """Show reused semantic content and its optional guarded fit."""
    def alpha_contour_points(rgba):
        arr = np.asarray(rgba, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 4:
            return None
        alpha = arr[:, :, 3]
        if int((alpha > 0).sum()) == 0:
            return None
        contours, _hier = cv2.findContours((alpha > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cnt is None or cnt.shape[0] < 3:
            return None
        return cnt[:, 0, :].astype(np.float64)

    display_rgba = np.asarray(img_rgba, dtype=np.uint8).copy()
    visible_mask = display_mask if display_mask is not None else source_mask
    if visible_mask is not None:
        visible_mask = np.asarray(visible_mask, dtype=bool)
        if visible_mask.shape == display_rgba.shape[:2]:
            display_rgba[~visible_mask, :3] = 0
            display_rgba[~visible_mask, 3] = 0

    rgba = Image.fromarray(display_rgba).convert("RGBA")
    base = Image.alpha_composite(
        Image.new("RGBA", rgba.size, (246, 246, 244, 255)),
        rgba
    )
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    poly = [tuple(map(float, p)) for p in np.asarray(wall_poly_px, dtype=np.float64)]
    if len(poly) >= 3:
        draw.line(poly + [poly[0]], fill=(255, 0, 0, 230), width=2)

    def draw_points_as_line(points, color, width=2):
        if points is None:
            return
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return
        pts = _sample_polyline_points(pts, max_points=900)
        coords = [tuple(map(float, p)) for p in pts]
        draw.line(coords + [coords[0]], fill=color, width=width)

    def contour_groups_from_mask(mask):
        if mask is None:
            return []
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != img_rgba.shape[:2] or not mask_arr.any():
            return []
        contours, _hier = cv2.findContours(
            mask_arr.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return [
            contour[:, 0, :].astype(np.float64)
            for contour in contours
            if contour is not None
            and contour.shape[0] >= 3
            and cv2.contourArea(contour) >= QUAD_MIN_CONTOUR_AREA_PX
        ]

    def draw_contour_groups(groups, color, width=2, max_points=900):
        valid_groups = [
            np.asarray(group, dtype=np.float64)
            for group in groups
            if group is not None
            and np.asarray(group).ndim == 2
            and np.asarray(group).shape[0] >= 2
        ]
        total_points = sum(group.shape[0] for group in valid_groups)
        for group in valid_groups:
            quota = max(
                3,
                int(round(max_points * group.shape[0] / max(total_points, 1))),
            )
            sampled = _sample_polyline_points(group, max_points=quota)
            coords = [tuple(map(float, point)) for point in sampled]
            draw.line(coords + [coords[0]], fill=color, width=width)

    source_contours = contour_groups_from_mask(source_mask)
    if source_contours:
        draw_contour_groups(source_contours, (255, 180, 0, 235), width=3)
    else:
        if source_pts is None:
            source_pts = alpha_contour_points(img_rgba)
        draw_points_as_line(source_pts, (255, 180, 0, 235), width=3)

    fit_applied = bool((fit_info or {}).get("applied", False))
    fitted_color = (0, 255, 0, 235) if fit_applied else (255, 0, 255, 235)
    fitted_contours = []
    src = np.asarray(source_pts, dtype=np.float64) if source_pts is not None else None
    dst = np.asarray(fitted_pts, dtype=np.float64) if fitted_pts is not None else None
    if (
        source_contours
        and src is not None
        and dst is not None
        and src.ndim == 2
        and dst.shape == src.shape
        and src.shape[0] >= 3
        and np.isfinite(src).all()
        and np.isfinite(dst).all()
    ):
        design = np.column_stack([src, np.ones(src.shape[0], dtype=np.float64)])
        affine_coeffs, _residuals, _rank, _singular = np.linalg.lstsq(
            design,
            dst,
            rcond=None,
        )
        fitted_contours = [
            np.column_stack([
                contour,
                np.ones(contour.shape[0], dtype=np.float64),
            ]) @ affine_coeffs
            for contour in source_contours
        ]
    if fitted_contours:
        draw_contour_groups(fitted_contours, fitted_color, width=2)
    else:
        draw_points_as_line(fitted_pts, fitted_color, width=2)

    reason = str((fit_info or {}).get("reason", "unknown"))
    status = "edge adjustment: APPLIED" if fit_applied else f"edge adjustment: NOT APPLIED ({reason})"
    text_box = draw.textbbox((8, 8), status)
    draw.rectangle(
        (text_box[0] - 4, text_box[1] - 3, text_box[2] + 4, text_box[3] + 3),
        fill=(0, 0, 0, 180),
    )
    draw.text((8, 8), status, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(out_path)


def _save_reused_prefit_semantic_overlay(
    *,
    img_rgba,
    wall_poly_px,
    content_mask,
    exclusion_mask,
    out_path,
    reuse_info=None,
):
    """Visualize the persisted full-image SAM3 mask in rectified coordinates."""
    rgba_array = np.asarray(img_rgba, dtype=np.uint8)
    if rgba_array.ndim != 3 or rgba_array.shape[2] != 4:
        raise ValueError("img_rgba must be an HxWx4 array.")
    height, width = rgba_array.shape[:2]
    content = np.asarray(content_mask, dtype=bool)
    exclusion = np.asarray(exclusion_mask, dtype=bool)
    if content.shape != (height, width) or exclusion.shape != (height, width):
        raise ValueError("Semantic overlay masks must match img_rgba.")

    # Use an opaque neutral base so transparent/excluded pixels remain easy to
    # distinguish in the contact sheet.
    visible_rgba = rgba_array.copy()
    visible_rgba[:, :, 3] = 255
    base = Image.fromarray(visible_rgba, mode="RGBA")
    overlay_array = np.zeros((height, width, 4), dtype=np.uint8)
    overlay_array[content] = (0, 220, 70, 78)
    overlay_array[exclusion] = (255, 40, 140, 118)
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    composed = Image.alpha_composite(base, overlay)

    draw = ImageDraw.Draw(composed, "RGBA")
    polygon = [
        (float(x), float(y))
        for x, y in np.asarray(wall_poly_px, dtype=np.float64).reshape(-1, 2)
        if np.isfinite([x, y]).all()
    ]
    if len(polygon) >= 3:
        draw.line(
            polygon + [polygon[0]],
            fill=(255, 0, 0, 235),
            width=2,
        )

    accepted = bool((reuse_info or {}).get("accepted_for_reuse", False))
    reason = str((reuse_info or {}).get("reason", "unknown"))
    lines = [
        (
            "reused full-image SAM3 evidence: "
            f"{'ACCEPTED' if accepted else 'PROJECTION FALLBACK'}"
        ),
        "green: retained building content | pink: excluded / LaMa hole",
        "red: fitted wall projection",
        "no second SAM3 inference; nearest-neighbor mask propagation",
    ]
    roof_removal = dict(
        (reuse_info or {}).get("post_hough_roof_structure_removal") or {}
    )
    if int(roof_removal.get("roof_pixels", 0)) > 0:
        lines.append(
            "post-Hough roof removal: "
            f"{int(roof_removal.get('roof_component_count', 0))} roof(s), "
            f"{int(roof_removal.get('divider_component_count', 0))} divider(s), "
            f"{int(roof_removal.get('removed_pixels', 0))} px removed"
        )
    if not accepted:
        lines.append(f"fallback reason: {reason}")

    try:
        text_boxes = [draw.textbbox((0, 0), line) for line in lines]
        text_height = max(12, max(box[3] - box[1] for box in text_boxes))
        text_width = max(box[2] - box[0] for box in text_boxes)
    except AttributeError:
        text_width = max(8 * len(line) for line in lines)
        text_height = 12
    panel_height = 8 + len(lines) * (text_height + 3)
    draw.rectangle(
        (4, 4, min(width - 4, text_width + 16), min(height - 4, panel_height)),
        fill=(0, 0, 0, 182),
    )
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255, 255))
        y += text_height + 3

    composed.convert("RGB").save(out_path)


def _facade_hough_edge_targets(wall_poly_px):
    poly = np.asarray(wall_poly_px, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return []
    finite = np.isfinite(poly).all(axis=1)
    poly = poly[finite]
    if poly.shape[0] < 3:
        return []
    if np.linalg.norm(poly[0] - poly[-1]) < 1e-6:
        poly = poly[:-1]
    if poly.shape[0] < 3:
        return []

    ymin = float(np.min(poly[:, 1]))
    ymax = float(np.max(poly[:, 1]))
    height = max(ymax - ymin, 1.0)
    targets = []
    min_edge_len = max(20.0, float(HOUGH_MIN_LENGTH_PX) * 0.35)

    for i in range(poly.shape[0]):
        p0 = poly[i].astype(np.float64)
        p1 = poly[(i + 1) % poly.shape[0]].astype(np.float64)
        length = float(np.linalg.norm(p1 - p0))
        if length < min_edge_len:
            continue

        angle = angle_deg_of_segment(p0, p1)
        horizontal_diff = angle_diff_deg_180(angle, 0.0)
        mid = 0.5 * (p0 + p1)
        is_bottom = (
            horizontal_diff <= 25.0 and
            float(mid[1]) >= ymin + 0.65 * height
        )
        targets.append({
            "edge_index": int(i),
            "target_p0": p0,
            "target_p1": p1,
            "length_px": length,
            "angle_deg": float(angle),
            "is_bottom": bool(is_bottom),
        })
    return targets

def _facade_hough_side_edge_targets(edge_targets):
    """
    Pick only the outer vertical-ish side edges for grouped facade Hough
    correction. Non-quad roof steps/slopes are useful geometry, but they are
    poor warp anchors and can distort the texture.
    """
    verticalish = []
    for edge in edge_targets:
        if bool(edge.get("is_bottom")):
            continue
        angle = float(edge.get("angle_deg", 0.0))
        if angle_diff_deg_180(angle, 90.0) > 35.0:
            continue

        target_p0 = np.asarray(edge["target_p0"], dtype=np.float64)
        target_p1 = np.asarray(edge["target_p1"], dtype=np.float64)
        mid = 0.5 * (target_p0 + target_p1)
        rec = dict(edge)
        rec["mid_x"] = float(mid[0])
        rec["mid_y"] = float(mid[1])
        verticalish.append(rec)

    if len(verticalish) < 2:
        return []

    left_edge = min(verticalish, key=lambda e: float(e["mid_x"]))
    right_edge = max(verticalish, key=lambda e: float(e["mid_x"]))
    if int(left_edge["edge_index"]) == int(right_edge["edge_index"]):
        return []

    left_edge = dict(left_edge)
    right_edge = dict(right_edge)
    left_edge["side"] = "left"
    right_edge["side"] = "right"
    return [left_edge, right_edge]

def _primary_hough_lines_from_edges(selected_edges):
    valid = [
        edge for edge in selected_edges
        if edge.get("selected_line") is not None
    ]
    if not valid:
        return None, None, None

    def mid(edge):
        return 0.5 * (
            np.asarray(edge["target_p0"], dtype=np.float64) +
            np.asarray(edge["target_p1"], dtype=np.float64)
        )

    verticalish = [
        edge for edge in valid
        if angle_diff_deg_180(float(edge.get("angle_deg", edge.get("target_angle_deg", 0.0))), 90.0) <= 35.0
    ]
    if len(verticalish) >= 2:
        left_edge = min(verticalish, key=lambda e: float(mid(e)[0]))
        right_edge = max(verticalish, key=lambda e: float(mid(e)[0]))
    else:
        left_edge = min(valid, key=lambda e: float(mid(e)[0]))
        right_edge = max(valid, key=lambda e: float(mid(e)[0]))

    top_candidates = [
        edge for edge in valid
        if not bool(edge.get("is_bottom"))
        and edge is not left_edge
        and edge is not right_edge
    ]
    top_edge = min(top_candidates, key=lambda e: float(mid(e)[1])) if top_candidates else None

    return (
        np.asarray(left_edge["selected_line"], dtype=np.float64) if left_edge else None,
        np.asarray(right_edge["selected_line"], dtype=np.float64) if right_edge else None,
        np.asarray(top_edge["selected_line"], dtype=np.float64) if top_edge else None,
    )

def _fit_alpha_boundary_line_for_target(edge_map_u8, target_p0, target_p1, search_band_u8, min_length_px, angle_thresh_deg):
    info = {
        "fallback": "alpha_boundary_point_fit",
        "num_boundary_points": 0,
        "best_length_px": None,
        "best_angle_diff_deg": None,
        "best_distance_px": None,
        "best_overlap_ratio": None,
    }
    target_p0 = np.asarray(target_p0, dtype=np.float64)
    target_p1 = np.asarray(target_p1, dtype=np.float64)
    ys, xs = np.where((edge_map_u8 > 0) & (search_band_u8 > 0))
    info["num_boundary_points"] = int(len(xs))
    if len(xs) < 12:
        return None, info

    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    v = np.array([float(vx), float(vy)], dtype=np.float64)
    v /= max(float(np.linalg.norm(v)), 1e-9)
    p = np.array([float(x0), float(y0)], dtype=np.float64)

    target_dir = target_p1 - target_p0
    target_dir /= max(float(np.linalg.norm(target_dir)), 1e-9)
    if float(np.dot(v, target_dir)) < 0.0:
        v = -v

    angle_diff = angle_diff_deg_180(angle_deg_of_segment(p, p + v), angle_deg_of_segment(target_p0, target_p1))
    info["best_angle_diff_deg"] = float(angle_diff)
    if angle_diff > angle_thresh_deg:
        return None, info

    scalars = (pts.astype(np.float64) - p) @ v
    tmin = float(np.min(scalars))
    tmax = float(np.max(scalars))
    selected_line = np.vstack([p + tmin * v, p + tmax * v]).astype(np.float64)
    length = float(np.linalg.norm(selected_line[1] - selected_line[0]))
    info["best_length_px"] = length
    if length < max(20.0, min_length_px * 0.5):
        return None, info

    fitted_mid = 0.5 * (selected_line[0] + selected_line[1])
    info["best_distance_px"] = float(point_line_distance(fitted_mid[0], fitted_mid[1], target_p0, target_p1))
    overlap_px, line_px = line_overlap_with_edge_map(selected_line, edge_map_u8, thickness=3)
    info["best_overlap_ratio"] = float(overlap_px / max(line_px, 1))
    return selected_line, info

def _apply_group_hough_adjustment_legacy(
    ortho_rgba,
    wall_poly_px,
    rect_poly_px,
    per_building_out,
    geojson_base,
    facade_tag,
    edge_mask_override=None,
    allow_guided_warp=True,
    auxiliary_masks=None,
):
    hough_info = {
        "enabled": bool(ENABLE_ORTHO_HOUGH_DEBUG),
        "method": "rectified_semantically_filtered_rgb_canny_side_edges",
        "pipeline_stage": (
            "after_prefit_mask_rectification_before_optional_ortho_fit"
        ),
        "total_segments_detected": 0,
        "left_line": None,
        "right_line": None,
        "top_line": None,
        "selected_edges": [],
        "left_info": {},
        "right_info": {},
        "top_info": {},
        "polygon_affine_warp": {},
        "guided_warp_enabled": bool(ENABLE_HOUGH_GUIDED_WARP and allow_guided_warp),
        "single_side_warp_enabled": bool(globals().get(
            "ENABLE_HOUGH_SINGLE_SIDE_WARP", True,
        )),
        "guided_warp_applied": False,
        "guided_warp_axes": [],
    }
    hough_overlay_path = None
    hough_warp_overlay_path = None
    hough_band_paths = {}
    transformed_auxiliary_masks = {}
    for name, value in dict(auxiliary_masks or {}).items():
        mask = np.asarray(value, dtype=bool)
        if mask.shape != ortho_rgba.shape[:2]:
            raise ValueError(
                f"Hough auxiliary mask {name!r} must match ortho image shape."
            )
        transformed_auxiliary_masks[str(name)] = mask.copy()

    if wall_poly_px is None or np.asarray(wall_poly_px).shape[0] < 3:
        hough_info["reason"] = "missing_wall_polygon"
        return (
            ortho_rgba,
            hough_info,
            hough_overlay_path,
            hough_warp_overlay_path,
            hough_band_paths,
            transformed_auxiliary_masks,
        )

    wall_poly_px = np.asarray(wall_poly_px, dtype=np.float64)
    wall_mask_bool = build_wall_region_mask(
        ortho_rgba.shape[0],
        ortho_rgba.shape[1],
        wall_poly_px
    ) > 0
    texture_alpha_mask = ortho_rgba[:, :, 3] > 0
    alpha_mask = texture_alpha_mask & wall_mask_bool
    edge_source = "rectified_reused_semantic_content_edges"
    if edge_mask_override is not None:
        candidate_edge_mask = np.asarray(edge_mask_override, dtype=bool)
        if candidate_edge_mask.shape == texture_alpha_mask.shape and candidate_edge_mask.any():
            alpha_mask = candidate_edge_mask & wall_mask_bool
            edge_source = "explicit_reused_semantic_mask_boundary"
            hough_info["method"] = "bounded_mask_boundary_houghlinesp"
            hough_info["pipeline_stage"] = "explicit_mask_override"
    ortho_rgba = ortho_rgba.copy()
    ortho_rgba[:, :, 3] = (texture_alpha_mask.astype(np.uint8) * 255)
    ortho_before_hough = ortho_rgba.copy()

    if edge_source == "rectified_reused_semantic_content_edges":
        hough_edge_map_u8 = build_edge_map_for_hough(
            ortho_rgba[:, :, :3],
            alpha_mask,
        )
        # Masking the rectified crop to the projected wall creates a strong,
        # artificial edge exactly on the target polygon. Remove it so Hough
        # has to select facade content edges rather than rediscovering the crop.
        crop_boundary_u8 = build_alpha_boundary_edge_map_for_hough(alpha_mask)
        hough_edge_map_u8[crop_boundary_u8 > 0] = 0
        hough_info["suppressed_projection_crop_boundary"] = True
    else:
        hough_edge_map_u8 = build_alpha_boundary_edge_map_for_hough(alpha_mask)
        hough_info["suppressed_projection_crop_boundary"] = False
    wall_roi_u8 = (wall_mask_bool.astype(np.uint8) * 255)
    roi_kernel_size = max(3, int(2 * HOUGH_SEARCH_BAND_PX + 1))
    roi_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (roi_kernel_size, roi_kernel_size))
    hough_roi_u8 = cv2.dilate(wall_roi_u8, roi_kernel, iterations=1)

    hough_lines = []
    hough_left_line = None
    hough_right_line = None
    hough_top_line = None
    hough_left_info = {}
    hough_right_info = {}
    hough_top_info = {}
    selected_edges = []

    if ENABLE_ORTHO_HOUGH_DEBUG:
        hough_lines = detect_hough_segments(
            hough_edge_map_u8,
            roi_mask=hough_roi_u8
        )
        hough_info["total_segments_detected"] = int(len(hough_lines))
        hough_info["hough_mask_area_px"] = int(alpha_mask.sum())
        hough_info["edge_source"] = edge_source

        edge_targets = _facade_hough_edge_targets(wall_poly_px)
        side_targets = _facade_hough_side_edge_targets(edge_targets)
        hough_info["target_edge_count"] = int(len(edge_targets))
        hough_info["candidate_mode"] = "left_right_vertical_side_edges_only"
        hough_info["side_target_edge_indices"] = [
            int(edge["edge_index"]) for edge in side_targets
        ]
        hough_info["skipped_bottom_edge_indices"] = [
            int(edge["edge_index"]) for edge in edge_targets
            if bool(edge.get("is_bottom"))
        ]
        if len(side_targets) < 2:
            hough_info["side_target_reason"] = "fewer_than_two_vertical_side_edges"

        band_specs = {}
        for edge in side_targets:
            target_p0 = np.asarray(edge["target_p0"], dtype=np.float64)
            target_p1 = np.asarray(edge["target_p1"], dtype=np.float64)
            band_u8 = build_line_search_band(
                ortho_rgba.shape[0], ortho_rgba.shape[1],
                target_p0, target_p1, wall_mask_bool, HOUGH_SEARCH_BAND_PX
            )
            selected_line, edge_info = select_best_hough_line_for_target(
                hough_lines, target_p0, target_p1, band_u8, hough_edge_map_u8,
                min_length_px=HOUGH_MIN_LENGTH_PX,
                angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG
            )
            if selected_line is None:
                fallback_line, fallback_info = _fit_alpha_boundary_line_for_target(
                    hough_edge_map_u8,
                    target_p0,
                    target_p1,
                    band_u8,
                    min_length_px=HOUGH_MIN_LENGTH_PX,
                    angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG,
                )
                edge_info = dict(edge_info or {})
                edge_info["alpha_boundary_fallback"] = fallback_info
                if fallback_line is not None:
                    selected_line = fallback_line
                    edge_info["fallback_applied"] = True
                else:
                    edge_info["fallback_applied"] = False
            edge_record = {
                "edge_index": int(edge["edge_index"]),
                "target_p0": target_p0,
                "target_p1": target_p1,
                "target_length_px": float(edge["length_px"]),
                "target_angle_deg": float(edge["angle_deg"]),
                "is_bottom": False,
                "side": edge.get("side"),
                "selected_line": selected_line,
                "info": edge_info,
            }
            selected_edges.append(edge_record)
            band_specs[f"edge_{int(edge['edge_index']):02d}"] = band_u8

        side_lines_for_debug = {
            edge.get("side"): np.asarray(edge["selected_line"], dtype=np.float64)
            for edge in selected_edges
            if edge.get("side") in {"left", "right"} and edge.get("selected_line") is not None
        }
        hough_left_line = side_lines_for_debug.get("left")
        hough_right_line = side_lines_for_debug.get("right")
        hough_top_line = None

        print(f"   Hough total segments on {facade_tag}: {len(hough_lines)}")
        print(f"   Hough selected facade edges on {facade_tag}: {sum(1 for e in selected_edges if e.get('selected_line') is not None)}/{len(selected_edges)}")

        if ENABLE_HOUGH_GUIDED_WARP and allow_guided_warp:
            target_by_side = {
                edge.get("side"): edge
                for edge in selected_edges
                if edge.get("side") in {"left", "right"}
            }
            selected_by_side = {
                edge.get("side"): edge
                for edge in selected_edges
                if edge.get("side") in {"left", "right"} and edge.get("selected_line") is not None
            }
            left_edge = selected_by_side.get("left")
            right_edge = selected_by_side.get("right")
            left_target = target_by_side.get("left")
            right_target = target_by_side.get("right")
            detected_side_count = int(left_edge is not None) + int(right_edge is not None)
            single_side_allowed = bool(globals().get(
                "ENABLE_HOUGH_SINGLE_SIDE_WARP", True,
            ))
            can_apply_side_warp = bool(
                left_target is not None
                and right_target is not None
                and (
                    detected_side_count == 2
                    or (single_side_allowed and detected_side_count == 1)
                )
            )
            if can_apply_side_warp:
                proj_left_line = np.vstack([
                    np.asarray(left_target["target_p0"], dtype=np.float64),
                    np.asarray(left_target["target_p1"], dtype=np.float64),
                ])
                proj_right_line = np.vstack([
                    np.asarray(right_target["target_p0"], dtype=np.float64),
                    np.asarray(right_target["target_p1"], dtype=np.float64),
                ])
                ortho_rgba = apply_hough_guided_ortho_warp(
                    ortho_rgba=ortho_rgba,
                    sel_left_line=(
                        np.asarray(left_edge["selected_line"], dtype=np.float64)
                        if left_edge is not None else None
                    ),
                    sel_right_line=(
                        np.asarray(right_edge["selected_line"], dtype=np.float64)
                        if right_edge is not None else None
                    ),
                    sel_top_line=None,
                    proj_left_line=proj_left_line,
                    proj_right_line=proj_right_line,
                    proj_top_line=None,
                )
                for mask_name, mask_value in list(
                    transformed_auxiliary_masks.items()
                ):
                    transformed_auxiliary_masks[mask_name] = (
                        apply_hough_guided_ortho_warp(
                            ortho_rgba=mask_value.astype(np.uint8) * 255,
                            sel_left_line=(
                                np.asarray(
                                    left_edge["selected_line"],
                                    dtype=np.float64,
                                )
                                if left_edge is not None else None
                            ),
                            sel_right_line=(
                                np.asarray(
                                    right_edge["selected_line"],
                                    dtype=np.float64,
                                )
                                if right_edge is not None else None
                            ),
                            sel_top_line=None,
                            proj_left_line=proj_left_line,
                            proj_right_line=proj_right_line,
                            proj_top_line=None,
                            interpolation=cv2.INTER_NEAREST,
                        ) > 0
                    )
                hough_info["guided_warp_applied"] = True
                hough_info["guided_warp_axes"] = [
                    f"{side}_edge_{int(edge['edge_index']):02d}"
                    for side, edge in (("left", left_edge), ("right", right_edge))
                    if edge is not None
                ]
                detected_sides = [
                    side
                    for side, edge in (("left", left_edge), ("right", right_edge))
                    if edge is not None
                ]
                identity_anchor_side = (
                    None
                    if detected_side_count == 2
                    else ("left" if left_edge is None else "right")
                )
                hough_info["side_horizontal_warp"] = {
                    "applied": True,
                    "method": (
                        "left_right_piecewise_horizontal"
                        if detected_side_count == 2
                        else "single_side_piecewise_horizontal_with_identity_anchor"
                    ),
                    "detected_sides": detected_sides,
                    "identity_anchor_side": identity_anchor_side,
                }
                hough_info["polygon_affine_warp"] = {
                    "applied": False,
                    "reason": "disabled_for_grouped_facades_side_edges_only",
                }
                print(f"   Hough side-edge horizontal warp applied on {facade_tag}: {hough_info['guided_warp_axes']}")
            else:
                if detected_side_count == 0:
                    skip_reason = "no_side_line_detected"
                elif detected_side_count == 1 and not single_side_allowed:
                    skip_reason = "single_side_warp_disabled"
                else:
                    skip_reason = "missing_side_target_geometry"
                hough_info["side_horizontal_warp"] = {
                    "applied": False,
                    "reason": skip_reason,
                }
                hough_info["polygon_affine_warp"] = {
                    "applied": False,
                    "reason": "disabled_for_grouped_facades_side_edges_only",
                }
                print(
                    f"   Hough side-edge horizontal warp skipped on {facade_tag} "
                    f"({skip_reason})"
                )

        outside_wall_after_warp = (
            (ortho_rgba[:, :, 3] > 0) & ~wall_mask_bool
        )
        hough_info["outside_wall_pixels_removed_after_warp"] = int(
            outside_wall_after_warp.sum()
        )
        hough_info["clipped_to_wall_projection_after_warp"] = True
        ortho_rgba[~wall_mask_bool, :3] = 0
        ortho_rgba[~wall_mask_bool, 3] = 0
        for mask_name in list(transformed_auxiliary_masks):
            transformed_auxiliary_masks[mask_name] &= wall_mask_bool

        if HOUGH_SAVE_BAND_MASKS:
            for band_name, band_u8 in band_specs.items():
                p = Path(per_building_out) / f"{geojson_base}__{facade_tag}__hough_{band_name}_band.png"
                Image.fromarray(band_u8 * 255).save(p)
                hough_band_paths[band_name] = p

    hough_overlay_path = Path(per_building_out) / f"{geojson_base}__{facade_tag}__hough_overlay.png"
    save_hough_all_lines_overlay(
        img_pil=Image.fromarray(ortho_before_hough).convert("RGBA"),
        wall_quad_xy=wall_poly_px,
        all_lines=hough_lines,
        selected_left=hough_left_line,
        selected_right=hough_right_line,
        selected_top=hough_top_line,
        out_path=str(hough_overlay_path),
        selected_edges=selected_edges,
    )

    if (
        ENABLE_HOUGH_GUIDED_WARP
        and allow_guided_warp
        and SAVE_HOUGH_WARP_DEBUG
        and hough_info.get("guided_warp_applied")
    ):
        hough_warp_overlay_path = Path(per_building_out) / f"{geojson_base}__{facade_tag}__hough_warp_overlay.png"
        save_hough_warp_overlay(
            img_pil=Image.fromarray(ortho_rgba).convert("RGBA"),
            wall_quad_xy=wall_poly_px,
            out_path=str(hough_warp_overlay_path)
        )

    hough_info.update({
        "left_line": [[float(x), float(y)] for x, y in hough_left_line.tolist()] if hough_left_line is not None else None,
        "right_line": [[float(x), float(y)] for x, y in hough_right_line.tolist()] if hough_right_line is not None else None,
        "top_line": [[float(x), float(y)] for x, y in hough_top_line.tolist()] if hough_top_line is not None else None,
        "selected_edges": [
            {
                "edge_index": int(edge["edge_index"]),
                "target_p0": [float(v) for v in np.asarray(edge["target_p0"], dtype=np.float64).tolist()],
                "target_p1": [float(v) for v in np.asarray(edge["target_p1"], dtype=np.float64).tolist()],
                "target_length_px": float(edge.get("target_length_px", 0.0)),
                "target_angle_deg": float(edge.get("target_angle_deg", 0.0)),
                "side": edge.get("side"),
                "selected_line": (
                    [[float(x), float(y)] for x, y in np.asarray(edge["selected_line"], dtype=np.float64).tolist()]
                    if edge.get("selected_line") is not None else None
                ),
                "info": edge.get("info", {}),
            }
            for edge in selected_edges
        ],
        "left_info": hough_left_info,
        "right_info": hough_right_info,
        "top_info": hough_top_info,
    })
    return (
        ortho_rgba,
        hough_info,
        hough_overlay_path,
        hough_warp_overlay_path,
        hough_band_paths,
        transformed_auxiliary_masks,
    )


def _save_opening_aware_rectification_overlay(
    rgba,
    wall_polygon,
    opening_info,
    out_path,
):
    """Draw corrected observed quads and their axis-aligned fitted targets."""
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bgr = cv2.cvtColor(
        np.asarray(Image.alpha_composite(white, image).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )
    polygon = np.rint(np.asarray(wall_polygon)).astype(np.int32)
    if len(polygon) >= 3:
        cv2.polylines(bgr, [polygon], True, (0, 0, 255), 2)
    homography = np.asarray(
        opening_info.get("homography", np.eye(3)), dtype=np.float64
    )
    drawn = 0
    for row in opening_info.get("openings", []):
        quad = np.asarray(
            row.get("quad_xy_tl_tr_br_bl", []), dtype=np.float64
        )
        if quad.shape != (4, 2):
            continue
        try:
            corrected = apply_H(quad, homography)
        except ValueError:
            continue
        left = float(np.mean(corrected[[0, 3], 0]))
        right = float(np.mean(corrected[[1, 2], 0]))
        top = float(np.mean(corrected[[0, 1], 1]))
        bottom = float(np.mean(corrected[[2, 3], 1]))
        fitted = np.asarray(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float64,
        )
        cv2.polylines(
            bgr,
            [np.rint(corrected).astype(np.int32)],
            True,
            (0, 220, 220),
            1,
        )
        cv2.polylines(
            bgr,
            [np.rint(fitted).astype(np.int32)],
            True,
            (0, 190, 0),
            2,
        )
        drawn += 1
    cv2.rectangle(bgr, (5, 5), (560, 55), (35, 35, 35), -1)
    cv2.putText(
        bgr,
        f"opening-aware: {drawn} accepted | yellow observed | green fitted",
        (14, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), bgr)


def _validated_side_extension_corridor(
    image_shape_hw,
    target_line,
    selected_line,
    candidate_extension_mask,
):
    """Keep only semantic content between one projected and observed side."""
    height, width = [int(value) for value in image_shape_hw]
    candidate = np.asarray(candidate_extension_mask, dtype=bool)
    if candidate.shape != (height, width) or not candidate.any():
        return np.zeros((height, width), dtype=bool)
    target = np.asarray(target_line, dtype=np.float64).reshape(2, 2)
    selected = np.asarray(selected_line, dtype=np.float64).reshape(2, 2)
    direction = selected[1] - selected[0]
    direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
    projected = []
    for point in target:
        along = float(np.dot(point - selected[0], direction))
        projected.append(selected[0] + along * direction)
    hull = cv2.convexHull(
        np.rint(np.vstack([target, np.asarray(projected)])).astype(np.int32)
    )
    corridor = np.zeros((height, width), dtype=np.uint8)
    if len(hull) >= 3:
        cv2.fillConvexPoly(corridor, hull, 1, lineType=cv2.LINE_8)
        corridor = cv2.dilate(
            corridor,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
    return candidate & (corridor > 0)


def _mask_rgba_to_semantic_content(rgba, content_mask):
    """Make the accepted semantic mask authoritative before resampling."""
    output = np.asarray(rgba, dtype=np.uint8).copy()
    keep = np.asarray(content_mask, dtype=bool)
    if keep.shape != output.shape[:2]:
        raise ValueError("Semantic content mask must match the RGBA image.")
    output[~keep, :3] = 0
    output[:, :, 3] = keep.astype(np.uint8) * 255
    return output


def _apply_group_hough_adjustment(
    ortho_rgba,
    wall_poly_px,
    rect_poly_px,
    per_building_out,
    geojson_base,
    facade_tag,
    edge_mask_override=None,
    allow_guided_warp=True,
    auxiliary_masks=None,
    side_evidence=None,
    opening_context=None,
):
    """Validate side evidence, then prefer one shared opening-aware warp."""
    del rect_poly_px  # Roof/base rectangle edges are not alignment evidence.
    rgba = np.asarray(ortho_rgba, dtype=np.uint8).copy()
    height, width = rgba.shape[:2]
    wall_poly = np.asarray(wall_poly_px, dtype=np.float64)
    transformed_masks = {}
    for name, value in dict(auxiliary_masks or {}).items():
        mask = np.asarray(value, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"Hough auxiliary mask {name!r} must match ortho image shape."
            )
        transformed_masks[str(name)] = mask.copy()

    hough_stage_enabled = bool(ENABLE_ORTHO_HOUGH_DEBUG)
    opening_stage_enabled = bool(globals().get(
        "ENABLE_OPENING_AWARE_RECTIFICATION", True,
    ))
    info = {
        "enabled": hough_stage_enabled,
        "method": "side_specific_semantic_first_validated_hough",
        "pipeline_stage": "pre_clip_joint_side_and_opening_rectification",
        "total_segments_detected": 0,
        "left_line": None,
        "right_line": None,
        "top_line": None,
        "selected_edges": [],
        "left_info": {},
        "right_info": {},
        "top_info": {},
        "guided_warp_enabled": bool(
            ENABLE_HOUGH_GUIDED_WARP and allow_guided_warp
        ),
        "single_side_warp_enabled": bool(globals().get(
            "ENABLE_HOUGH_SINGLE_SIDE_WARP", True,
        )),
        "guided_warp_applied": False,
        "guided_warp_axes": [],
        "accepted_side_extension_pixels": 0,
        "accepted_side_extension_sides": [],
        "opening_aware": {
            "enabled": opening_stage_enabled,
            "applied": False,
            "reason": "not_run",
        },
        "side_evidence": side_evidence_metadata(side_evidence or {}),
    }
    hough_overlay_path = None
    hough_warp_overlay_path = None
    hough_band_paths = {}
    if wall_poly.ndim != 2 or wall_poly.shape[0] < 3:
        info["reason"] = "missing_wall_polygon"
        return (
            rgba, info, hough_overlay_path, hough_warp_overlay_path,
            hough_band_paths, transformed_masks,
        )

    wall_mask = build_wall_region_mask(height, width, wall_poly) > 0
    texture_mask = rgba[:, :, 3] > 0
    before = rgba.copy()
    if edge_mask_override is not None:
        explicit = np.asarray(edge_mask_override, dtype=bool)
        if explicit.shape != (height, width):
            raise ValueError("Explicit Hough edge mask must match ortho image.")
        rgb_edges = build_alpha_boundary_edge_map_for_hough(explicit)
        info["edge_source"] = "explicit_boundary_override"
    else:
        rgb_edges = build_edge_map_for_hough(rgba[:, :, :3], texture_mask)
        # Suppress only the synthetic projected polygon boundary.  The old
        # implementation erased the complete semantic/alpha outline, including
        # the true wall-to-sky edge requested as the preferred cue.
        projected_boundary = build_alpha_boundary_edge_map_for_hough(wall_mask)
        rgb_edges[projected_boundary > 0] = 0
        info["edge_source"] = "rgb_canny_with_projection_boundary_suppressed"

    side_rows = dict((side_evidence or {}).get("sides") or {})
    preferred_union = np.zeros((height, width), dtype=bool)
    for row in side_rows.values():
        for key in ("preferred_inside_mask", "preferred_outside_mask"):
            mask = np.asarray(
                row.get(key, np.zeros((height, width), dtype=bool)),
                dtype=bool,
            )
            if mask.shape == (height, width):
                preferred_union |= mask
    rgb_edges[preferred_union] = 255

    all_lines = []
    selected_edges = []
    band_specs = {}
    edge_targets = _facade_hough_edge_targets(wall_poly)
    all_side_targets = _facade_hough_side_edge_targets(edge_targets)
    side_targets = all_side_targets if hough_stage_enabled else []
    info.update({
        "target_edge_count": int(len(edge_targets)),
        "candidate_mode": "left_right_independent_inside_then_outside",
        "side_target_edge_indices": [
            int(edge["edge_index"]) for edge in all_side_targets
        ],
        "skipped_bottom_edge_indices": [
            int(edge["edge_index"])
            for edge in edge_targets if bool(edge.get("is_bottom"))
        ],
    })
    if not hough_stage_enabled:
        info["hough_detection_reason"] = "disabled_by_configuration"

    for edge in side_targets:
        side = str(edge.get("side"))
        target_p0 = np.asarray(edge["target_p0"], dtype=np.float64)
        target_p1 = np.asarray(edge["target_p1"], dtype=np.float64)
        side_row = dict(side_rows.get(side) or {})
        default_band = build_line_search_band(
            height, width, target_p0, target_p1, wall_mask,
            int(HOUGH_SEARCH_BAND_PX),
        ) > 0
        attempts = []
        if bool(side_row.get("major_foreground_occlusion", False)):
            attempts = []
            edge_info = {
                "rejection_reason": "major_foreground_occlusion_preserve_current_crop",
                "foreground_occlusion_fraction": float(
                    side_row.get("foreground_occlusion_fraction", 0.0)
                ),
                "attempts": [],
            }
            selected_line = None
            selected_attempt = None
        else:
            if side_row:
                attempt_specs = [
                    ("inside_semantic", "inside_search_mask", "preferred_inside_mask", True),
                    ("inside_rgb", "inside_search_mask", None, True),
                    ("outside_semantic", "outside_search_mask", "preferred_outside_mask", True),
                    (
                        "outside_rgb", "outside_search_mask", None,
                        bool(side_row.get("outside_rgb_allowed", False)),
                    ),
                ]
            else:
                attempt_specs = [("legacy_band_rgb", None, None, True)]
            selected_line = None
            selected_attempt = None
            edge_info = {"attempts": []}
            for attempt_name, band_key, preferred_key, enabled in attempt_specs:
                if not enabled:
                    continue
                band = default_band if band_key is None else np.asarray(
                    side_row.get(
                        band_key, np.zeros((height, width), dtype=bool)
                    ),
                    dtype=bool,
                )
                if band.shape != (height, width) or not band.any():
                    edge_info["attempts"].append({
                        "name": attempt_name,
                        "accepted": False,
                        "reason": "empty_search_region",
                    })
                    continue
                preferred = None
                if preferred_key is not None:
                    preferred = np.asarray(
                        side_row.get(
                            preferred_key,
                            np.zeros((height, width), dtype=bool),
                        ),
                        dtype=bool,
                    )
                    if preferred.shape != (height, width) or not preferred.any():
                        edge_info["attempts"].append({
                            "name": attempt_name,
                            "accepted": False,
                            "reason": "no_semantic_interface_in_search_region",
                        })
                        continue
                    attempt_edges = preferred.astype(np.uint8) * 255
                else:
                    attempt_edges = rgb_edges
                lines = detect_hough_segments(
                    attempt_edges, roi_mask=band.astype(np.uint8) * 255
                )
                all_lines.extend(lines)
                maximum_side_distance_px = min(
                    float(globals().get(
                        "HOUGH_SIDE_MAX_DISTANCE_PX", 36.0,
                    )),
                    max(
                        6.0,
                        float(edge["length_px"])
                        * float(globals().get(
                            "HOUGH_SIDE_MAX_DISTANCE_TARGET_RATIO", 0.04,
                        )),
                    ),
                )
                candidate, candidate_info = select_best_hough_line_for_target(
                    lines,
                    target_p0,
                    target_p1,
                    band.astype(np.uint8),
                    attempt_edges,
                    min_length_px=float(HOUGH_MIN_LENGTH_PX),
                    angle_thresh_deg=float(globals().get(
                        "HOUGH_SIDE_ANGLE_THRESH_DEG", 8.0,
                    )),
                    minimum_target_coverage_ratio=float(globals().get(
                        "HOUGH_SIDE_MIN_TARGET_COVERAGE_RATIO", 0.75,
                    )),
                    maximum_length_ratio=float(globals().get(
                        "HOUGH_SIDE_MAX_LENGTH_RATIO", 1.20,
                    )),
                    maximum_distance_px=maximum_side_distance_px,
                    minimum_band_occupancy_ratio=float(globals().get(
                        "HOUGH_SIDE_MIN_BAND_OCCUPANCY_RATIO", 0.80,
                    )),
                    minimum_edge_support_ratio=float(globals().get(
                        (
                            "FACADE_SIDE_MIN_SEMANTIC_INTERFACE_SUPPORT"
                            if preferred is not None
                            else "HOUGH_SIDE_MIN_EDGE_SUPPORT_RATIO"
                        ),
                        0.20 if preferred is not None else 0.30,
                    )),
                    offset_cluster_px=float(globals().get(
                        "HOUGH_SIDE_OFFSET_CLUSTER_PX", 12.0,
                    )),
                    preferred_edge_map_u8=(
                        preferred.astype(np.uint8) * 255
                        if preferred is not None else None
                    ),
                )
                attempt_record = {
                    "name": attempt_name,
                    "accepted": candidate is not None,
                    "segment_count": int(len(lines)),
                    "info": candidate_info,
                }
                edge_info["attempts"].append(attempt_record)
                band_specs[f"{side}_{attempt_name}"] = band.astype(np.uint8)
                if candidate is not None:
                    selected_line = candidate
                    selected_attempt = attempt_name
                    edge_info.update(candidate_info)
                    edge_info["selected_attempt"] = attempt_name
                    edge_info["selection_source"] = (
                        "semantic_interface_hough"
                        if preferred is not None else "rgb_canny_hough"
                    )
                    break
            if selected_line is None:
                edge_info["rejection_reason"] = (
                    "no_inside_or_permitted_outside_candidate_passed_all_hard_gates"
                )

        if selected_attempt == "inside_semantic":
            side_decision = "inside_edge_preferred_semantic"
        elif selected_attempt == "inside_rgb":
            side_decision = "inside_edge_rgb"
        elif selected_attempt == "outside_semantic":
            side_decision = (
                "outside_adjacent_edge_semantic"
                if bool(side_row.get("adjacent_visible", False))
                else "background_interface_semantic"
            )
        elif selected_attempt == "outside_rgb":
            side_decision = "outside_adjacent_edge_rgb"
        elif bool(side_row.get("major_foreground_occlusion", False)):
            side_decision = "keep_current_occlusion"
        else:
            side_decision = "no_safe_edge"
        edge_info["side_decision"] = side_decision
        edge_record = {
            "edge_index": int(edge["edge_index"]),
            "target_p0": target_p0,
            "target_p1": target_p1,
            "target_length_px": float(edge["length_px"]),
            "target_angle_deg": float(edge["angle_deg"]),
            "is_bottom": False,
            "side": side,
            "selected_line": selected_line,
            "info": edge_info,
        }
        selected_edges.append(edge_record)
        if side == "left":
            info["left_info"] = edge_info
        elif side == "right":
            info["right_info"] = edge_info

    opening_context = dict(opening_context or {})
    rectified_side_extensions = {}
    source_side_extensions = {}
    source_extension_rows = dict(
        opening_context.get("source_side_extensions") or {}
    )
    source_to_rectified_h = opening_context.get("source_to_rectified_h")
    inverse_source_h = None
    if source_to_rectified_h is not None:
        try:
            inverse_source_h = np.linalg.inv(
                np.asarray(source_to_rectified_h, dtype=np.float64)
            )
        except np.linalg.LinAlgError:
            inverse_source_h = None
    for edge in selected_edges:
        side = str(edge.get("side"))
        attempt = str((edge.get("info") or {}).get("selected_attempt", ""))
        if (
            edge.get("selected_line") is None
            or not attempt.startswith("outside_")
        ):
            continue
        row = dict(side_rows.get(side) or {})
        rectified_candidate = np.asarray(
            row.get(
                "candidate_extension_mask",
                np.zeros((height, width), dtype=bool),
            ),
            dtype=bool,
        )
        accepted_rectified = _validated_side_extension_corridor(
            (height, width),
            np.vstack([edge["target_p0"], edge["target_p1"]]),
            edge["selected_line"],
            rectified_candidate,
        )
        rectified_side_extensions[side] = accepted_rectified
        source_candidate = source_extension_rows.get(side)
        if source_candidate is not None and inverse_source_h is not None:
            source_candidate = np.asarray(source_candidate, dtype=bool)
            source_height, source_width = source_candidate.shape
            source_corridor = cv2.warpPerspective(
                accepted_rectified.astype(np.uint8),
                inverse_source_h,
                (source_width, source_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            source_side_extensions[side] = source_candidate & source_corridor

    info["total_segments_detected"] = int(len(all_lines))
    info["hough_mask_area_px"] = int(texture_mask.sum())
    info["validated_outside_extension_candidates"] = {
        side: int(mask.sum())
        for side, mask in rectified_side_extensions.items()
    }
    opening_authoritative = False
    opening_applied = False
    if (
        opening_stage_enabled
        and opening_context.get("source_rows") is not None
    ):
        opening_authoritative = bool(
            len(opening_context.get("source_rows", [])) > 0
        )
        try:
            residual_h, opening_info, _observations = (
                estimate_opening_aware_rectification(
                    opening_context.get("source_rows", []),
                    np.asarray(
                        opening_context["source_wall_mask"], dtype=bool
                    ),
                    np.asarray(
                        opening_context["source_to_rectified_h"],
                        dtype=np.float64,
                    ),
                    (height, width),
                    wall_mask,
                    wall_poly,
                    selected_edges,
                    source_exclusion_mask=(
                        None
                        if opening_context.get("source_exclusion_mask") is None
                        else np.asarray(
                            opening_context["source_exclusion_mask"],
                            dtype=bool,
                        )
                    ),
                    minimum_sam_score=float(globals().get(
                        "OPENING_AWARE_MIN_SAM_SCORE", 0.25,
                    )),
                    minimum_stability=float(globals().get(
                        "OPENING_AWARE_MIN_STABILITY", 0.78,
                    )),
                    minimum_openings=int(globals().get(
                        "OPENING_AWARE_MIN_OPENINGS", 3,
                    )),
                    maximum_side_consensus_deg=float(globals().get(
                        "OPENING_AWARE_MAX_CONSENSUS_DEG", 5.0,
                    )),
                    allow_projective=bool(globals().get(
                        "OPENING_AWARE_ALLOW_PROJECTIVE", True,
                    )),
                    maximum_final_side_angle_deg=float(globals().get(
                        "OPENING_AWARE_MAX_FINAL_SIDE_ANGLE_DEG", 2.0,
                    )),
                    maximum_final_side_distance_px=float(globals().get(
                        "OPENING_AWARE_MAX_FINAL_SIDE_DISTANCE_PX", 8.0,
                    )),
                    maximum_final_opening_p90_axis_error_deg=float(
                        globals().get(
                            "OPENING_AWARE_MAX_FINAL_P90_AXIS_ERROR_DEG",
                            3.0,
                        )
                    ),
                    maximum_final_opening_p90_orthogonality_error_deg=float(
                        globals().get(
                            "OPENING_AWARE_MAX_FINAL_P90_ORTHOGONALITY_ERROR_DEG",
                            5.0,
                        )
                    ),
                    maximum_final_per_opening_axis_error_deg=float(
                        globals().get(
                            "OPENING_AWARE_MAX_FINAL_PER_OPENING_AXIS_ERROR_DEG",
                            4.0,
                        )
                    ),
                    maximum_final_per_opening_orthogonality_error_deg=float(
                        globals().get(
                            "OPENING_AWARE_MAX_FINAL_PER_OPENING_ORTHOGONALITY_ERROR_DEG",
                            5.0,
                        )
                    ),
                )
            )
            info["opening_aware"] = opening_info
            for constraint_row in opening_info.get("side_constraints", []):
                if constraint_row.get("rejection_reason") != (
                    "conflicts_with_opening_consensus"
                ):
                    continue
                conflict_side = str(constraint_row.get("side"))
                for selected_edge in selected_edges:
                    if str(selected_edge.get("side")) != conflict_side:
                        continue
                    selected_edge["selected_line"] = None
                    selected_edge_info = selected_edge.setdefault(
                        "info", {}
                    )
                    selected_edge_info[
                        "opening_consensus_veto"
                    ] = True
                    selected_edge_info[
                        "rejection_reason"
                    ] = "conflicts_with_opening_consensus"
            opening_authoritative = bool(
                int(opening_info.get("accepted_opening_count", 0))
                >= int(globals().get("OPENING_AWARE_MIN_OPENINGS", 3))
            )
            opening_applied = bool(opening_info.get("applied", False))
            if opening_applied:
                admitted_sides = {
                    str(row.get("side"))
                    for row in opening_info.get("side_constraints", [])
                    if bool(row.get("admitted", False))
                }
                source_rgba = opening_context.get("source_rgba")
                source_masks = {
                    str(name): np.asarray(mask, dtype=bool).copy()
                    for name, mask in dict(
                        opening_context.get("source_masks") or {}
                    ).items()
                }
                accepted_source_extension = np.zeros(
                    np.asarray(source_rgba).shape[:2]
                    if source_rgba is not None
                    else (1, 1),
                    dtype=bool,
                )
                for side in admitted_sides:
                    extension = source_side_extensions.get(side)
                    if (
                        extension is not None
                        and extension.shape == accepted_source_extension.shape
                    ):
                        accepted_source_extension |= extension
                for mask_name in ("semantic_content", "semantic_candidate"):
                    if (
                        mask_name in source_masks
                        and source_masks[mask_name].shape
                        == accepted_source_extension.shape
                    ):
                        source_masks[mask_name] |= accepted_source_extension
                if source_rgba is not None and source_masks:
                    total_h = residual_h @ np.asarray(
                        opening_context["source_to_rectified_h"],
                        dtype=np.float64,
                    )
                    render_source_rgba = np.asarray(
                        source_rgba, dtype=np.uint8
                    )
                    if "semantic_content" in source_masks:
                        # Candidate outside-wall pixels were available for line
                        # detection, but only a validated side corridor may be
                        # sampled into the final texture.
                        render_source_rgba = _mask_rgba_to_semantic_content(
                            render_source_rgba,
                            source_masks["semantic_content"],
                        )
                    rgba = cv2.warpPerspective(
                        render_source_rgba,
                        total_h,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0),
                    )
                    for mask_name, source_mask in source_masks.items():
                        transformed_masks[str(mask_name)] = cv2.warpPerspective(
                            np.asarray(source_mask, dtype=np.uint8),
                            total_h,
                            (width, height),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        ) > 0
                    info["opening_aware"]["rendering"] = {
                        "method": "one_pass_from_source",
                        "H_source_to_final_rectified": total_h.tolist(),
                        "rgb_interpolation": "linear",
                        "mask_interpolation": "nearest",
                    }
                else:
                    accepted_rectified_extension = np.zeros(
                        (height, width), dtype=bool
                    )
                    for side in admitted_sides:
                        accepted_rectified_extension |= rectified_side_extensions.get(
                            side, np.zeros((height, width), dtype=bool)
                        )
                    for mask_name in ("semantic_content", "semantic_candidate"):
                        if mask_name in transformed_masks:
                            transformed_masks[mask_name] |= (
                                accepted_rectified_extension
                            )
                    if "semantic_content" in transformed_masks:
                        rgba = _mask_rgba_to_semantic_content(
                            rgba, transformed_masks["semantic_content"]
                        )
                    rgba = cv2.warpPerspective(
                        rgba,
                        residual_h,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0),
                    )
                    for mask_name, mask in list(transformed_masks.items()):
                        transformed_masks[mask_name] = cv2.warpPerspective(
                            mask.astype(np.uint8),
                            residual_h,
                            (width, height),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        ) > 0
                info["guided_warp_applied"] = True
                info["accepted_side_extension_pixels"] = int(
                    accepted_source_extension.sum()
                    if source_rgba is not None
                    else accepted_rectified_extension.sum()
                )
                info["accepted_side_extension_sides"] = sorted(
                    admitted_sides
                )
                info["guided_warp_axes"] = [
                    "opening_vertical_family",
                    "opening_horizontal_family",
                ] + [
                    f"{row['side']}_validated_side"
                    for row in opening_info.get("side_constraints", [])
                    if bool(row.get("admitted", False))
                ]
                info["side_horizontal_warp"] = {
                    "applied": False,
                    "reason": "replaced_by_shared_opening_aware_homography",
                }
        except Exception as opening_exc:
            info["opening_aware"] = {
                "enabled": True,
                "applied": False,
                "reason": f"opening_solver_failed: {opening_exc}",
            }
            print(f"[{facade_tag}] opening-aware rectification failed: {opening_exc}")

    if (
        not opening_applied
        and not opening_authoritative
        and hough_stage_enabled
        and ENABLE_HOUGH_GUIDED_WARP
        and allow_guided_warp
    ):
        target_by_side = {
            edge["side"]: edge for edge in selected_edges
            if edge.get("side") in {"left", "right"}
        }
        selected_by_side = {
            side: edge for side, edge in target_by_side.items()
            if edge.get("selected_line") is not None
        }
        left_target = target_by_side.get("left")
        right_target = target_by_side.get("right")
        left_edge = selected_by_side.get("left")
        right_edge = selected_by_side.get("right")
        count = int(left_edge is not None) + int(right_edge is not None)
        single_allowed = bool(globals().get(
            "ENABLE_HOUGH_SINGLE_SIDE_WARP", True,
        ))
        can_warp = bool(
            left_target is not None
            and right_target is not None
            and (count == 2 or (single_allowed and count == 1))
        )
        if can_warp:
            accepted_rectified_extension = np.zeros(
                (height, width), dtype=bool
            )
            accepted_extension_sides = []
            for side, selected_edge in (
                ("left", left_edge), ("right", right_edge)
            ):
                if selected_edge is None:
                    continue
                extension = rectified_side_extensions.get(side)
                if extension is not None:
                    accepted_rectified_extension |= extension
                    if extension.any():
                        accepted_extension_sides.append(side)
            for mask_name in ("semantic_content", "semantic_candidate"):
                if mask_name in transformed_masks:
                    transformed_masks[mask_name] |= (
                        accepted_rectified_extension
                    )
            if "semantic_content" in transformed_masks:
                rgba = _mask_rgba_to_semantic_content(
                    rgba, transformed_masks["semantic_content"]
                )
            projected_left = np.vstack([
                left_target["target_p0"], left_target["target_p1"]
            ])
            projected_right = np.vstack([
                right_target["target_p0"], right_target["target_p1"]
            ])
            selected_left = (
                None if left_edge is None else left_edge["selected_line"]
            )
            selected_right = (
                None if right_edge is None else right_edge["selected_line"]
            )
            rgba = apply_hough_guided_ortho_warp(
                rgba,
                selected_left,
                selected_right,
                None,
                projected_left,
                projected_right,
                None,
            )
            for mask_name, mask in list(transformed_masks.items()):
                transformed_masks[mask_name] = apply_hough_guided_ortho_warp(
                    mask.astype(np.uint8),
                    selected_left,
                    selected_right,
                    None,
                    projected_left,
                    projected_right,
                    None,
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            info["guided_warp_applied"] = True
            info["accepted_side_extension_pixels"] = int(
                accepted_rectified_extension.sum()
            )
            info["accepted_side_extension_sides"] = sorted(
                accepted_extension_sides
            )
            info["guided_warp_axes"] = [
                f"{side}_edge_{int(edge['edge_index']):02d}"
                for side, edge in (("left", left_edge), ("right", right_edge))
                if edge is not None
            ]
            info["side_horizontal_warp"] = {
                "applied": True,
                "method": (
                    "validated_two_side_piecewise_horizontal"
                    if count == 2
                    else "validated_single_side_piecewise_with_identity_anchor"
                ),
            }
        else:
            info["side_horizontal_warp"] = {
                "applied": False,
                "reason": "no_validated_side_configuration",
            }
    elif opening_authoritative and not opening_applied:
        info["side_horizontal_warp"] = {
            "applied": False,
            "reason": "reliable_openings_present_identity_fallback_blocks_legacy_side_warp",
        }

    if "semantic_content" in transformed_masks:
        final_content = np.asarray(
            transformed_masks["semantic_content"], dtype=bool
        )
        if final_content.shape == (height, width):
            info["nonaccepted_candidate_pixels_removed"] = int(
                ((rgba[:, :, 3] > 0) & (~final_content)).sum()
            )
            rgba = _mask_rgba_to_semantic_content(rgba, final_content)

    outside = (rgba[:, :, 3] > 0) & (~wall_mask)
    info["outside_wall_pixels_removed_after_warp"] = int(outside.sum())
    info["clipped_to_wall_projection_after_warp"] = True
    rgba[~wall_mask, :3] = 0
    rgba[~wall_mask, 3] = 0
    for mask_name in list(transformed_masks):
        transformed_masks[mask_name] &= wall_mask

    selected_by_side = {
        edge.get("side"): edge for edge in selected_edges
        if edge.get("selected_line") is not None
    }
    left_line = (
        None
        if selected_by_side.get("left") is None
        else np.asarray(selected_by_side["left"]["selected_line"])
    )
    right_line = (
        None
        if selected_by_side.get("right") is None
        else np.asarray(selected_by_side["right"]["selected_line"])
    )
    hough_overlay_path = Path(
        per_building_out,
        f"{geojson_base}__{facade_tag}__hough_overlay.png",
    )
    save_hough_all_lines_overlay(
        img_pil=Image.fromarray(before).convert("RGBA"),
        wall_quad_xy=wall_poly,
        all_lines=all_lines,
        selected_left=left_line,
        selected_right=right_line,
        selected_top=None,
        out_path=str(hough_overlay_path),
        selected_edges=selected_edges,
    )
    if (
        bool(globals().get("SAVE_OPENING_AWARE_DEBUG", True))
        and (info.get("opening_aware") or {}).get("openings")
    ):
        opening_overlay_path = Path(
            per_building_out,
            f"{geojson_base}__{facade_tag}__opening_aware_overlay.png",
        )
        _save_opening_aware_rectification_overlay(
            rgba,
            wall_poly,
            info["opening_aware"],
            opening_overlay_path,
        )
        info["opening_aware"]["overlay_png"] = (
            opening_overlay_path.name
        )
    if (
        info["guided_warp_applied"]
        and SAVE_HOUGH_WARP_DEBUG
        and allow_guided_warp
    ):
        hough_warp_overlay_path = Path(
            per_building_out,
            f"{geojson_base}__{facade_tag}__hough_warp_overlay.png",
        )
        save_hough_warp_overlay(
            Image.fromarray(rgba).convert("RGBA"),
            wall_poly,
            str(hough_warp_overlay_path),
        )
    if HOUGH_SAVE_BAND_MASKS:
        for band_name, band in band_specs.items():
            path = Path(
                per_building_out,
                f"{geojson_base}__{facade_tag}__hough_{band_name}_band.png",
            )
            Image.fromarray(band.astype(np.uint8) * 255).save(path)
            hough_band_paths[band_name] = path

    info.update({
        "left_line": None if left_line is None else left_line.tolist(),
        "right_line": None if right_line is None else right_line.tolist(),
        "selected_edges": [
            {
                "edge_index": int(edge["edge_index"]),
                "target_p0": np.asarray(edge["target_p0"]).tolist(),
                "target_p1": np.asarray(edge["target_p1"]).tolist(),
                "target_length_px": float(edge["target_length_px"]),
                "target_angle_deg": float(edge["target_angle_deg"]),
                "side": edge.get("side"),
                "selected_line": (
                    None
                    if edge.get("selected_line") is None
                    else np.asarray(edge["selected_line"]).tolist()
                ),
                "info": edge.get("info", {}),
            }
            for edge in selected_edges
        ],
    })
    return (
        rgba,
        info,
        hough_overlay_path,
        hough_warp_overlay_path,
        hough_band_paths,
        transformed_masks,
    )

def _safe_artifact_folder_part(value):
    text = str(value)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text).strip("_") or "item"

def _wall_group_projection_export_enabled():
    return bool(globals().get("SAVE_TEMP_GLOBAL_WALL_GROUP_IMAGE_PROJECTIONS", False))

def _wall_group_projection_export_staging_root(per_building_out):
    out_dir = Path(per_building_out)
    export_name = str(globals().get(
        "TEMP_GLOBAL_WALL_GROUP_IMAGE_EXPORT_FOLDER",
        "wall_group_image_projections",
    ))
    if not bool(globals().get("SAVE_WALL_ARTIFACT_FOLDERS", True)):
        return out_dir / "global" / export_name
    staging_name = str(globals().get(
        "TEMP_GLOBAL_WALL_GROUP_IMAGE_STAGING_FOLDER",
        "_tmp_wall_group_image_projections",
    ))
    return out_dir / staging_name

def _save_wall_group_outline_projection_image(
    img_pil,
    outline_uv_px,
    out_path,
    segment_indices=None,
):
    base = img_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    pts = [
        (float(x), float(y))
        for x, y in np.asarray(outline_uv_px, dtype=np.float64).reshape(-1, 2)
        if np.isfinite([x, y]).all()
    ]
    if len(pts) >= 2:
        edges = (
            [(int(a), int(b)) for a, b in segment_indices]
            if segment_indices is not None
            else [(i, (i + 1) % len(pts)) for i in range(len(pts))]
        )
        for index0, index1 in edges:
            if 0 <= index0 < len(pts) and 0 <= index1 < len(pts):
                draw.line([pts[index0], pts[index1]], fill=(255, 0, 0, 255), width=3)
        for x, y in pts:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255, 235), outline=(255, 0, 0, 255))
    Image.alpha_composite(base, overlay).convert("RGB").save(out_path)

def _render_model_depth_view(
    *,
    meshes_named,
    K,
    R_wc,
    C,
    source_image_size,
    output_image_size=None,
    image_to_output_H=None,
):
    if not meshes_named:
        return None
    source_image_size = tuple(int(v) for v in source_image_size)
    output_image_size = (
        tuple(int(v) for v in output_image_size)
        if output_image_size is not None else source_image_size
    )
    depth = render_model_depth_map(
        meshes_named,
        K,
        R_wc,
        C,
        source_image_size,
        near_m=float(globals().get("MODEL_DEPTH_NEAR_M", 0.05)),
    )
    if image_to_output_H is not None:
        depth = warp_depth_map_to_canvas(depth, image_to_output_H, output_image_size)
    return depth

def _save_model_depth_map_artifacts(
    *,
    per_building_out,
    prefix_name,
    meshes_named,
    K,
    R_wc,
    C,
    source_image_size,
    output_image_size=None,
    image_to_output_H=None,
    camera_metadata=None,
    precomputed_output_depth=None,
):
    if not bool(globals().get("SAVE_MODEL_DEPTH_MAPS", True)):
        return {}
    if not meshes_named and precomputed_output_depth is None:
        return {}

    source_image_size = tuple(int(v) for v in source_image_size)
    output_image_size = (
        tuple(int(v) for v in output_image_size)
        if output_image_size is not None else source_image_size
    )

    transform_used = image_to_output_H is not None
    if precomputed_output_depth is None:
        depth = _render_model_depth_view(
            meshes_named=meshes_named,
            K=K,
            R_wc=R_wc,
            C=C,
            source_image_size=source_image_size,
            output_image_size=output_image_size,
            image_to_output_H=image_to_output_H,
        )
    else:
        depth = np.asarray(precomputed_output_depth, dtype=np.float32)
        expected_shape = (int(output_image_size[1]), int(output_image_size[0]))
        if depth.shape != expected_shape:
            raise ValueError(
                f"Precomputed depth shape {depth.shape} does not match output canvas {expected_shape}."
            )

    out_root = Path(per_building_out)
    prefix = out_root / prefix_name
    npy_path = prefix.with_suffix(".npy")
    png16_path = prefix.with_name(prefix.name + "_mm_u16.png")
    visual_path = prefix.with_name(prefix.name + "_visual.png")
    meta_path = prefix.with_name(prefix.name + "_meta.json")

    np.save(npy_path, depth.astype(np.float32))
    Image.fromarray(depth_map_to_uint16_mm(depth)).save(png16_path)
    Image.fromarray(depth_map_to_visual_png(depth)).save(visual_path)

    valid = np.isfinite(depth) & (depth > 0)
    metadata = {
        "type": "model_depth_map",
        "depth_units": "meters_camera_forward_z",
        "invalid_value": "NaN in npy; 0 in uint16 millimeter PNG",
        "source_image_size_px": [int(source_image_size[0]), int(source_image_size[1])],
        "output_image_size_px": [int(output_image_size[0]), int(output_image_size[1])],
        "image_to_output_H": (
            [[float(v) for v in row] for row in np.asarray(image_to_output_H, dtype=np.float64).tolist()]
            if transform_used else None
        ),
        "valid_pixel_count": int(valid.sum()),
        "min_depth_m": float(np.nanmin(depth[valid])) if valid.any() else None,
        "max_depth_m": float(np.nanmax(depth[valid])) if valid.any() else None,
        "mean_depth_m": float(np.nanmean(depth[valid])) if valid.any() else None,
        "near_m": float(globals().get("MODEL_DEPTH_NEAR_M", 0.05)),
        "uint16_png_max_mm": int(globals().get("MODEL_DEPTH_MAX_MM_PNG", 65535)),
        "camera": camera_metadata or {},
        "artifacts": {
            "depth_npy": npy_path.name,
            "depth_mm_u16_png": png16_path.name,
            "depth_visual_png": visual_path.name,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"   Saved model depth map: {visual_path.name}")
    return {
        "depth_npy": npy_path.name,
        "depth_mm_u16_png": png16_path.name,
        "depth_visual_png": visual_path.name,
        "depth_meta_json": meta_path.name,
        "valid_pixel_count": int(valid.sum()),
    }

def _save_temp_global_wall_group_image_projection(
    *,
    per_building_out,
    geojson_base,
    facade_tag,
    group_id,
    group_records,
    outline_xyz,
    raw_sources,
):
    if not _wall_group_projection_export_enabled():
        return []

    outline_xyz = np.asarray(outline_xyz, dtype=np.float64)
    if outline_xyz.ndim != 2 or outline_xyz.shape[0] < 3 or outline_xyz.shape[1] != 3 or not np.isfinite(outline_xyz).all():
        return []

    raw_sources = list(raw_sources or [])
    if not raw_sources:
        return []

    root = _wall_group_projection_export_staging_root(per_building_out)
    group_dir = root / f"{geojson_base}__{_safe_artifact_folder_part(facade_tag)}"
    group_dir.mkdir(parents=True, exist_ok=True)

    saved_dirs = []

    for source_idx, src in enumerate(raw_sources):
        if src is None or src.get("img") is None:
            continue
        src_img = src["img"].convert("RGB")
        src_rec = src.get("rec", {}) or {}
        pano_id = str(src_rec.get("pano_id", "unknown"))
        heading = float(src.get("heading", 0.0))
        pitch = float(src.get("pitch", 0.0))
        fov_deg = float(src.get("fov", 0.0))
        source_name = (
            f"source_{int(source_idx):02d}"
            f"__pano_{_safe_artifact_folder_part(pano_id)}"
            f"__hdg_{int(round(heading))}"
            f"__pit_{int(round(pitch))}"
            f"__fov_{int(round(fov_deg))}"
        )
        image_dir = group_dir / source_name
        image_dir.mkdir(parents=True, exist_ok=True)

        streetview_name = "streetview.jpg"
        projection_name = "wall_projection.png"
        streetview_path = image_dir / streetview_name
        projection_path = image_dir / projection_name

        src_img.save(streetview_path, quality=95)

        outline_uv_px, outline_edge_indices, _world_points, _projection_info = (
            project_outline_world_edges_near_clipped(
                outline_xyz,
                src["K"],
                src["Rwc"],
                src["C"],
                near_m=FACADE_PROJECTION_NEAR_PLANE_M,
            )
        )
        _save_wall_group_outline_projection_image(
            src_img,
            outline_uv_px,
            projection_path,
            segment_indices=outline_edge_indices,
        )

        saved_dirs.append(image_dir)

    return saved_dirs

def _move_if_file(src, dst_dir):
    src = Path(src)
    if not src.is_file():
        return None
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return dst

def _artifact_wall_index_from_name(name):
    text = str(name)
    for pattern in (r"_wall(\d+)", r"\bwall_(\d+)", r"_w(\d+)(?:\D|$)"):
        m = re.search(pattern, text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _mark_selected_candidate_overlay(output_path):
    """Add a compact check badge to the selected raw candidate overlay."""
    path = Path(output_path)
    if not path.is_file():
        return

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    diameter = int(np.clip(round(min(width, height) * 0.075), 28, 48))
    margin = max(8, diameter // 4)
    left = width - margin - diameter
    top = margin
    right = left + diameter
    bottom = top + diameter

    draw = ImageDraw.Draw(image)
    stroke = max(3, diameter // 10)
    draw.ellipse(
        (left, top, right, bottom),
        fill=(22, 163, 74),
        outline=(255, 255, 255),
        width=max(2, stroke // 2),
    )
    draw.line(
        [
            (left + 0.24 * diameter, top + 0.53 * diameter),
            (left + 0.43 * diameter, top + 0.72 * diameter),
            (left + 0.78 * diameter, top + 0.30 * diameter),
        ],
        fill=(255, 255, 255),
        width=stroke,
        joint="curve",
    )
    image.save(path)


def _save_candidate_projection_screening_overlay(source, raw_outline_px, output_path):
    """Show guidance, whole-model fit evidence, and the fitted target wall."""
    image_rgb = np.asarray(source["img"].convert("RGB"), dtype=np.uint8)
    candidate_guidance = source.get(
        "depth_global_fit_semantic_guidance",
        source.get("depth_global_prefit_semantic_guidance"),
    )
    semantic_guidance_drawn = False
    if isinstance(candidate_guidance, dict):
        try:
            image_rgb = create_prefit_semantic_guidance_overlay(
                image_rgb,
                candidate_guidance,
                # The raw model is drawn below from real projected model edges.
                # Re-contouring this mask would recreate viewport closures.
                draw_raw_projection_outline=False,
                draw_legend=False,
            )
            semantic_guidance_drawn = True
        except Exception as semantic_overlay_exc:
            source["depth_global_combined_overlay_semantic_error"] = (
                f"{type(semantic_overlay_exc).__name__}: "
                f"{semantic_overlay_exc}"
            )
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    projection_H = np.asarray(
        source.get("selection_projection_H", np.eye(3)),
        dtype=np.float64,
    )

    obstruction_mask = source.get("external_building_occlusion_mask")
    if obstruction_mask is not None:
        obstruction_mask = np.asarray(obstruction_mask, dtype=bool)
        if obstruction_mask.shape == (height, width) and obstruction_mask.any():
            obstruction_contours, _ = cv2.findContours(
                obstruction_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            for contour in obstruction_contours:
                points = np.asarray(contour[:, 0, :], dtype=np.int32)
                for index in range(len(points)):
                    point0 = points[index]
                    point1 = points[(index + 1) % len(points)]
                    point0_on_frame = bool(
                        int(point0[0]) in (0, width - 1)
                        or int(point0[1]) in (0, height - 1)
                    )
                    point1_on_frame = bool(
                        int(point1[0]) in (0, width - 1)
                        or int(point1[1]) in (0, height - 1)
                    )
                    if point0_on_frame and point1_on_frame:
                        continue
                    draw_styled_line(
                        canvas,
                        point0,
                        point1,
                        OSM_OBSTRUCTION_LINE,
                        color_space="bgr",
                    )

    def finite_points(value):
        try:
            points = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return np.zeros((0, 2), dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            return np.zeros((0, 2), dtype=np.float64)
        return points

    fit_result_value = source.get("depth_global_fit_result")
    fit_result = (
        fit_result_value
        if isinstance(fit_result_value, dict)
        else {}
    )
    fitted_target_wall_points = finite_points(
        fit_result.get("depth_global_fitted_wall_outline_px", [])
    )
    real_model_points = finite_points(
        fit_result.get("fit_original_points", [])
    )
    real_fitted_model_points = finite_points(
        fit_result.get("fit_fitted_points", [])
    )
    real_model_segments = [
        (int(index0), int(index1))
        for index0, index1 in (fit_result.get("fit_segment_indices") or [])
    ]
    use_real_model_edges = bool(
        fit_result.get("fit_geometry_source")
        == "visible_semantic_projected_edges"
        and real_model_points.shape[0] >= 2
        and real_model_segments
    )
    if use_real_model_edges:
        raw_model_points = real_model_points
        fitted_model_points = real_fitted_model_points
        model_segments = real_model_segments
        model_geometry_source = "visible_real_whole_model_edges"
        frame_wrappers_already_filtered = True
    else:
        raw_model_points = finite_points(fit_result.get("original_points", []))
        fitted_model_points = finite_points(
            fit_result.get("fitted_points", [])
        )
        model_segments = [
            (int(index0), int(index1))
            for index0, index1 in (fit_result.get("segment_indices") or [])
        ]
        model_geometry_source = "filtered_whole_model_depth_silhouette"
        frame_wrappers_already_filtered = bool(
            fit_result.get("depth_frame_wrappers_filtered", False)
        )

    used_depth_fallback = False
    if raw_model_points.shape[0] < 2 or not model_segments:
        full_model_depth = source.get("depth_global_full_model_depth")
        try:
            fallback_geometry = extract_depth_silhouette_geometry(
                np.asarray(full_model_depth, dtype=np.float32),
                minimum_area_px=int(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_MIN_AREA_PX",
                    350,
                )),
                minimum_component_fraction=float(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_MIN_COMPONENT_FRACTION",
                    0.02,
                )),
                contour_epsilon_px=float(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_CONTOUR_EPSILON_PX",
                    1.5,
                )),
                maximum_points=int(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_MAX_POINTS",
                    240,
                )),
            )
            raw_model_points = finite_points(fallback_geometry["points"])
            model_segments = [
                (int(index0), int(index1))
                for index0, index1 in fallback_geometry["segment_indices"]
            ]
            used_depth_fallback = True
            frame_wrappers_already_filtered = bool(
                fallback_geometry.get("frame_wrappers_filtered", False)
            )
            model_geometry_source = (
                "filtered_whole_model_depth_silhouette"
            )
        except Exception:
            raw_model_points = np.zeros((0, 2), dtype=np.float64)
            model_segments = []

    fit_applied = bool(
        source.get(
            "depth_global_fit_applied",
            fit_result.get("applied", False),
        )
    )
    if fit_applied and fitted_target_wall_points.shape[0] < 3:
        raw_target_wall_points = finite_points(
            source.get("selection_visible_wall_outline_px", raw_outline_px)
        )
        if raw_target_wall_points.shape[0] >= 3:
            fitted_target_wall_points = apply_H(
                raw_target_wall_points,
                projection_H,
            )
    if fit_applied and (
        used_depth_fallback
        or fitted_model_points.shape != raw_model_points.shape
    ):
        fitted_model_points = apply_H(raw_model_points, projection_H)
    excluded_border_segment_indices = []
    if not frame_wrappers_already_filtered:
        model_segments, excluded_border_segment_indices = (
            filter_image_border_wrapper_segments(
                raw_model_points,
                model_segments,
                (height, width),
                epsilon_px=float(globals().get(
                    "MODEL_DEPTH_BOUNDARY_IMAGE_BORDER_EPSILON_PX",
                    0.5,
                )),
            )
        )
    source["depth_global_projection_overlay_excluded_frame_segment_count"] = (
        int(len(excluded_border_segment_indices))
    )
    source["depth_global_projection_overlay_geometry"] = model_geometry_source

    def draw_model_segments(
        points,
        segments,
        style: OverlayLineStyle,
    ):
        points = finite_points(points)
        drawn = False
        for raw_index0, raw_index1 in segments:
            index0 = int(raw_index0)
            index1 = int(raw_index1)
            if not (
                0 <= index0 < len(points)
                and 0 <= index1 < len(points)
            ):
                continue
            endpoints = points[[index0, index1]]
            if not np.isfinite(endpoints).all():
                continue
            coordinate_limit = 1_000_000_000.0
            point0 = tuple(
                np.round(np.clip(endpoints[0], -coordinate_limit, coordinate_limit))
                .astype(np.int64)
                .tolist()
            )
            point1 = tuple(
                np.round(np.clip(endpoints[1], -coordinate_limit, coordinate_limit))
                .astype(np.int64)
                .tolist()
            )
            visible, clipped0, clipped1 = cv2.clipLine(
                (0, 0, width, height),
                point0,
                point1,
            )
            if not visible:
                continue
            draw_styled_line(
                canvas,
                clipped0,
                clipped1,
                style,
                color_space="bgr",
            )
            drawn = True
        return drawn

    raw_model_drawn = draw_model_segments(
        raw_model_points,
        model_segments,
        RAW_MODEL_LINE,
    )
    fitted_model_drawn = bool(
        fit_applied
        and draw_model_segments(
            fitted_model_points,
            model_segments,
            ACCEPTED_MODEL_LINE,
        )
    )
    fitted_target_wall_drawn = bool(
        fit_applied
        and fitted_target_wall_points.shape[0] >= 3
        and draw_model_segments(
            fitted_target_wall_points,
            [
                (index, (index + 1) % fitted_target_wall_points.shape[0])
                for index in range(fitted_target_wall_points.shape[0])
            ],
            ACCEPTED_MODEL_LINE,
        )
    )

    if raw_model_drawn:
        geometry_line = model_projection_legend(fitted=fitted_model_drawn)
    elif fitted_target_wall_drawn:
        geometry_line = (
            "whole-model projection unavailable; dashed magenta=fitted "
            "target-wall projection"
        )
    else:
        geometry_line = (
            "whole-model projection unavailable; target-wall outline is hidden"
        )

    fit_status = (
        "accepted"
        if fit_applied
        else "raw fallback"
    )
    blocked = source.get("external_building_occlusion_fraction")
    blocked_text = (
        f" | OSM blocked {100.0 * float(blocked):.2f}%"
        if source.get("external_building_occlusion_available", False)
        else " | OSM unavailable"
    )
    transform = dict(fit_result.get("transform", {}))
    transform_text = (
        f" | scale={float(transform.get('scale', 1.0)):.4f}"
        f" tx={float(transform.get('tx', 0.0)):.1f}px"
        f" ty={float(transform.get('ty', 0.0)):.1f}px"
        f" gain={float(fit_result.get('score_improvement', 0.0)):.4f}"
        if fit_result
        else ""
    )
    lines = []
    if semantic_guidance_drawn:
        lines.extend(
            BACKGROUND_AWARE_SEMANTIC_LEGEND_ROWS
            if bool(candidate_guidance.get("background_aware_active", False))
            else SEMANTIC_LEGEND_ROWS
        )
        if isinstance(
            candidate_guidance.get("strict_roof_diagnostic_masks"),
            dict,
        ):
            lines.append(STRICT_ROOF_AUDIT_LEGEND_ROW)
    lines.extend([
        geometry_line,
        OSM_LEGEND_ROW,
        *([SEARCH_LEGEND_ROW] if semantic_guidance_drawn else []),
        f"depth-global {fit_status}{transform_text}{blocked_text}",
    ])
    draw_legend(canvas, lines, color_space="bgr")
    cv2.imwrite(str(output_path), canvas)


def _external_building_removal_preview_rgb(image_rgb, remove_mask):
    """Return a checkerboard preview so removed obstruction pixels are visible."""
    image = np.asarray(image_rgb.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(remove_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("External-building removal mask does not match the source image.")
    height, width = mask.shape
    yy, xx = np.indices((height, width))
    checker = np.where(((xx // 14 + yy // 14) % 2)[..., None], 213, 238).astype(np.uint8)
    checker = np.repeat(checker, 3, axis=2)
    preview = image.copy()
    preview[mask] = checker[mask]
    return preview


def _save_external_building_removal_preview(image_rgb, remove_mask, output_path):
    """Save a checkerboard preview so transparent obstruction holes are visible."""
    preview = _external_building_removal_preview_rgb(image_rgb, remove_mask)
    Image.fromarray(preview, mode="RGB").save(output_path)


def _include_artifact_in_contact_sheet(path):
    n = Path(path).name.lower()
    if "post_rectification_sam3" in n:
        # Retired artifact from the older two-inference pipeline. Ignore stale
        # files if an output folder is reused.
        return False
    if "target_model_visibility" in n:
        return False
    if "model_depth_mm_u16" in n:
        return False
    if "model_depth_boundary_candidate_mask" in n:
        return False
    if "selected_depth_global_processing_overlay" in n:
        return False
    if "source_pano" in n and "prefit_semantic_guidance" in n:
        # Kept as a full-resolution forensic artifact. The contact sheet uses
        # the combined candidate card so SAM guidance and fit cannot appear out
        # of execution order or be mistaken for separate fitting passes.
        return False
    if "source_pano" in n:
        return bool(
            n.endswith("_overlay.png")
            and "wireframe_fit" not in n
        )
    if n.startswith("sv__") and "selected_native_source" in n:
        return False
    if "wall_fit_comparison_overlay" in n:
        return False
    if "wireframe_corrected_processing_overlay" in n:
        return False
    return True


def _artifact_stage_label_and_rank(path):
    name = Path(path).name
    n = name.lower()
    m_src = re.search(r"source_pano(\d+)", n)
    source_idx = int(m_src.group(1)) if m_src else 999
    if "source_pano" in n and "prefit_semantic_guidance" in n:
        return (
            100 + source_idx,
            (
                f"01 candidate {source_idx:02d}: standalone SAM3 guidance "
                "(off contact sheet)"
            ),
        )
    if "source_pano" in n and "overlay" in n:
        return (
            100 + source_idx,
            f"01 candidate {source_idx:02d}: SAM3-guided whole-model global fit",
        )
    if n.startswith("sv__") or "__sv_" in n:
        return 300, "02 selected native processing image"
    if "selected_source_wireframe_fit" in n:
        return 350, "02 selected wall-only wireframe correction"
    if "wireframe_fit" in n:
        return 350, "02 wall-only wireframe correction"
    if "model_depth_visual" in n:
        return 370, "03a selected source: raw depth before selected fit/refit"
    if "selected_external_building_removal_mask" in n:
        return 372, "03b OSM exclusion mask (white pixels ignored by refit)"
    if "selected_source_external_buildings_removed" in n:
        return 374, "03c OSM preview only (checkerboard; canvas not cropped)"
    if "model_depth_prefit_semantic_guidance" in n:
        return 375, "03d full-image SAM3 guidance before selected fit/refit"
    if "selected_depth_global_fit_evidence_preview" in n:
        return 376, "03e ACTUAL refit evidence (checkerboard pixels ignored)"
    if "model_depth_boundary_fit_overlay" in n or "whole_model_depth_boundary_fit" in n:
        return 377, "03f raw + fitted whole-model projection using 03e evidence"
    if "model_depth_silhouette_mask" in n:
        return 378, "03g raw silhouette + dashed fitted boundary shift"
    if "selected_wall_only_processing_overlay" in n:
        return 390, "04 selected wall-only wall projection (downstream)"
    if "model_depth" in n:
        return 370, "03a raw whole-model depth from selected camera"
    if "atlas_weight" in n:
        return 360, "03 atlas blend weights"
    if "depth_aware_region_fit" in n:
        return 750, "legacy disabled segmentation-driven depth shift"
    if "wireframe_corrected_processing_overlay" in n or "raw_overlay" in n:
        return 395, "04a wall-only processing projection (comparison)"
    if n.endswith("_overlay.png") and not any(k in n for k in ("sam3", "ortho", "hough", "polygon", "lr")):
        return 400, "04 processing wall projection overlay"
    if "lr_band" in n or "lr_overlay" in n:
        return 500, "05 side crop / LR band"
    if "projection_cropped_facade" in n:
        return (
            550,
            "06 fitted wall crop + reused full-image SAM3 content mask",
        )
    if "ortho_prefit" in n:
        return 600, "07 reused full-image SAM3 mask after rectification"
    if "reused_prefit_semantic_mask_after_hough" in n:
        return (
            800,
            (
                "09 reused full-image SAM3 mask after rectification + "
                "Hough (no new inference)"
            ),
        )
    if "sam3_instances" in n:
        return 720, "legacy perspective SAM instances"
    if "sam3" in n:
        return 730, "legacy perspective SAM selection"
    if "hough_overlay" in n:
        return (
            700,
            "08a bounded Hough lines on propagated semantic content",
        )
    if "hough_warp" in n:
        return (
            710,
            "08b bounded Hough warp + semantic-mask propagation",
        )
    if "hough_" in n and "_band" in n:
        return 720, "08 Hough search band"
    if "polygon_fit" in n or "ortho_fit" in n:
        return 900, "10 guarded rectified edge adjustment"
    if "lama_mask" in n:
        return 1100, "11 LaMa hole mask"
    if "ortho_overlay" in n or "ortho_final_rgba__overlay" in n:
        return 1300, "13 final texture overlay"
    if n.endswith("_ortho.png") or "ortho_final_rgba.png" in n:
        return 1200, "12 final rectified texture"
    if "debug" in n:
        return 1400, "debug"
    return 2000, "other artifact"

def _pil_resample_lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")

def _draw_label(draw, xy, text, fill=(25, 25, 25, 255), bg=(255, 255, 255, 215)):
    text = str(text)
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
    except AttributeError:
        tw, th = draw.textsize(text)
        bbox = (x, y, x + tw, y + th)
    draw.rectangle([bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3], fill=bg)
    draw.text((x, y), text, fill=fill)

def _truncate_text(text, limit=62):
    text = str(text)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."

def _debug_model_view_basis(view_rows=None):
    # Stage-00 is a model index view, not a simulation of the selected Street
    # View camera. Its orientation must remain identical across wall groups.
    del view_rows
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_dir = np.asarray(
        globals().get(
            "CONTACT_SHEET_MODEL_VIEW_DIRECTION",
            (0.9, -1.0, -0.75),
        ),
        dtype=np.float64,
    )
    if camera_dir.shape != (3,) or not np.isfinite(camera_dir).all():
        camera_dir = np.array([0.9, -1.0, -0.75], dtype=np.float64)
    camera_dir = camera_dir / max(float(np.linalg.norm(camera_dir)), 1e-9)
    right = np.cross(camera_dir, world_up)
    if float(np.linalg.norm(right)) < 1e-9:
        camera_dir = np.array([0.9, -1.0, -0.75], dtype=np.float64)
        camera_dir = camera_dir / float(np.linalg.norm(camera_dir))
        right = np.cross(camera_dir, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, camera_dir)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    return camera_dir, right, up

def _make_debug_model_projector(fit_points, width, height, pad=28, view_basis=None):
    fit_pts = np.asarray(fit_points, dtype=np.float64)
    if fit_pts.ndim != 2 or fit_pts.shape[0] == 0:
        return None
    _camera_dir, right, up = view_basis or _debug_model_view_basis()
    center = np.nanmean(fit_pts, axis=0)
    fit_rel = fit_pts - center
    fit_xy = np.column_stack([fit_rel @ right, -(fit_rel @ up)])
    mn = np.nanmin(fit_xy, axis=0)
    mx = np.nanmax(fit_xy, axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])

    def project(points):
        pts = np.asarray(points, dtype=np.float64)
        rel = pts - center
        xy = np.column_stack([rel @ right, -(rel @ up)])
        out = np.empty_like(xy)
        out[:, 0] = pad + (xy[:, 0] - mn[0]) * scale
        out[:, 1] = pad + (xy[:, 1] - mn[1]) * scale
        return out

    return project

def _project_debug_model_points(points, width, height, pad=28, view_basis=None):
    projector = _make_debug_model_projector(points, width, height, pad=pad, view_basis=view_basis)
    return None if projector is None else projector(points)

def _debug_model_face_depth(face_xyz, view_basis=None):
    camera_dir, _right, _up = view_basis or _debug_model_view_basis()
    pts = np.asarray(face_xyz, dtype=np.float64)
    center = np.nanmean(pts, axis=0)
    return float(center @ camera_dir)

def _debug_model_camera_point(view_rows=None):
    cams = []
    for row in view_rows or []:
        cam = row.get("camera_utm_xyz")
        if cam is None:
            continue
        cam = np.asarray(cam, dtype=np.float64)
        if cam.shape == (3,) and np.isfinite(cam).all():
            cams.append(cam)
    if not cams:
        return None
    return np.nanmean(np.vstack(cams), axis=0)

def _clamp_debug_point(pt, width, height, pad=10):
    x = min(max(float(pt[0]), pad), width - pad)
    y = min(max(float(pt[1]), pad), height - pad)
    return np.array([x, y], dtype=np.float64)

def _make_model_highlight_panel(all_rows, target_wall_indices, title, size=(420, 300), target_rows=None):
    W, H = size
    panel = Image.new("RGB", size, (248, 248, 246))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle([0, 0, W - 1, H - 1], outline=(185, 185, 185, 255), width=1)

    faces = []
    all_points = []
    target_set = set(int(i) for i in target_wall_indices if i is not None)
    for row in all_rows or []:
        q = row.get("wall_quad_xyz_b1b2t2t1")
        if not q:
            continue
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (4, 3) or not np.isfinite(arr).all():
            continue
        try:
            idx = int(row.get("global_index", row.get("loop_index", -1)))
        except (TypeError, ValueError):
            idx = -1
        faces.append((idx, arr))
        all_points.append(arr)

    draw.text((12, 10), _truncate_text(title, 52), fill=(20, 20, 20, 255))
    if not all_points:
        draw.text((18, 130), "No wall geometry in viewer_index", fill=(90, 90, 90, 255))
        return panel

    view_rows = list(target_rows or [])
    if not view_rows:
        for row in all_rows or []:
            try:
                idx = int(row.get("global_index", row.get("loop_index", -1)))
            except (TypeError, ValueError):
                continue
            if idx in target_set:
                view_rows.append(row)

    pts_3d = np.vstack(all_points)
    view_basis = _debug_model_view_basis(view_rows)
    projector = _make_debug_model_projector(pts_3d, W, H - 38, pad=30, view_basis=view_basis)
    projected = None if projector is None else projector(pts_3d)
    if projected is None:
        return panel

    point_lookup = {}
    cursor = 0
    for idx, arr in faces:
        point_lookup[idx] = projected[cursor: cursor + arr.shape[0]]
        cursor += arr.shape[0]

    for idx, arr in sorted(faces, key=lambda item: _debug_model_face_depth(item[1], view_basis), reverse=True):
        poly = [tuple(p) for p in point_lookup[idx]]
        is_target = idx in target_set
        fill = (235, 235, 231, 190) if not is_target else (238, 57, 70, 225)
        outline = (120, 120, 120, 220) if not is_target else (120, 20, 25, 255)
        draw.polygon(poly, fill=fill, outline=outline)
        if is_target:
            cx = float(np.mean([p[0] for p in poly]))
            cy = float(np.mean([p[1] for p in poly]))
            _draw_label(draw, (cx - 8, cy - 6), f"w{idx}", fill=(30, 30, 30, 255))

    camera_point = _debug_model_camera_point(view_rows)
    if camera_point is not None:
        camera_px_raw = projector(np.asarray([camera_point], dtype=np.float64))[0]
        camera_px = _clamp_debug_point(camera_px_raw, W, H - 38, pad=18)
        target_faces = [arr for idx, arr in faces if idx in target_set]
        if target_faces:
            target_center = np.nanmean(np.vstack(target_faces).reshape(-1, 3), axis=0)
            target_px = projector(np.asarray([target_center], dtype=np.float64))[0]
            _draw_dashed_line(
                draw,
                tuple(camera_px.tolist()),
                tuple(target_px.tolist()),
                fill=(30, 120, 230, 210),
                width=2,
                dash=7,
                gap=5,
            )
        r = 5
        draw.ellipse(
            [camera_px[0] - r, camera_px[1] - r, camera_px[0] + r, camera_px[1] + r],
            fill=(30, 120, 230, 255),
            outline=(10, 55, 135, 255),
            width=2,
        )
        _draw_label(
            draw,
            (camera_px[0] + 8, camera_px[1] - 8),
            "SV camera +2.5m",
            fill=(10, 55, 135, 255),
            bg=(255, 255, 255, 225),
        )

    legend_x = W - 175
    legend_y = H - 72
    draw.rectangle([legend_x, legend_y, W - 10, H - 10], fill=(255, 255, 255, 225), outline=(185, 185, 185, 255))
    draw.rectangle([legend_x + 10, legend_y + 12, legend_x + 28, legend_y + 28], fill=(238, 57, 70, 225), outline=(120, 20, 25, 255))
    legend_text = "highlighted group" if target_set else "no wall highlight"
    draw.text((legend_x + 36, legend_y + 11), legend_text, fill=(25, 25, 25, 255))
    walls_text = (
        "walls " + ",".join(str(i) for i in sorted(target_set))
        if target_set else "no wall indices"
    )
    draw.text((legend_x + 10, legend_y + 32), _truncate_text(walls_text, 28), fill=(65, 65, 65, 255))
    if camera_point is not None:
        cy = legend_y + 52
        draw.ellipse([legend_x + 12, cy - 5, legend_x + 22, cy + 5], fill=(30, 120, 230, 255), outline=(10, 55, 135, 255))
        draw.text((legend_x + 36, cy - 8), "SV camera point", fill=(25, 25, 25, 255))
    return panel

def _make_artifact_card(img, title, subtitle, size=(420, 330)):
    W, H = size
    card = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(card, "RGBA")
    draw.rectangle([0, 0, W - 1, H - 1], outline=(205, 205, 205, 255), width=1)
    draw.text((12, 10), _truncate_text(title, 54), fill=(20, 20, 20, 255))
    draw.text((12, 28), _truncate_text(subtitle, 66), fill=(80, 80, 80, 255))

    box = (12, 52, W - 12, H - 12)
    max_w = box[2] - box[0]
    max_h = box[3] - box[1]
    im = img.convert("RGBA")
    bg = Image.new("RGBA", im.size, (246, 246, 244, 255))
    im = Image.alpha_composite(bg, im)
    im.thumbnail((max_w, max_h), _pil_resample_lanczos())
    x = box[0] + (max_w - im.width) // 2
    y = box[1] + (max_h - im.height) // 2
    card.paste(im.convert("RGB"), (x, y))
    return card

def _save_artifact_contact_sheet(group_dir, folder_name, rows, all_rows, artifact_unit="facade_group"):
    if not SAVE_ARTIFACT_CONTACT_SHEET:
        return

    group_dir = Path(group_dir)
    if not group_dir.is_dir():
        return

    image_paths = [
        p for p in group_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and p.name != "debug_contact_sheet.png"
        and _include_artifact_in_contact_sheet(p)
    ]
    image_paths.sort(key=lambda p: (*_artifact_stage_label_and_rank(p), p.name.lower()))

    target_wall_indices = set()
    for row in rows or []:
        try:
            target_wall_indices.add(int(row.get("global_index", row.get("loop_index", -1))))
        except (TypeError, ValueError):
            pass
    if not target_wall_indices:
        for p in image_paths:
            idx = _artifact_wall_index_from_name(p.name)
            if idx is not None:
                target_wall_indices.add(idx)

    title = f"{folder_name} - {artifact_unit}"
    cards = [
        _make_artifact_card(
            _make_model_highlight_panel(
                all_rows,
                sorted(target_wall_indices),
                "untextured model highlight",
                target_rows=rows,
            ),
            "00 untextured model + highlighted group",
            "red faces are the wall fragments represented by this folder",
        )
    ]

    for p in image_paths:
        try:
            with Image.open(p) as im:
                rank, label = _artifact_stage_label_and_rank(p)
                cards.append(_make_artifact_card(im.copy(), label, p.name))
        except Exception:
            continue

    if not cards:
        return

    cols = 3
    card_w, card_h = (420, 330)
    gap = 18
    margin = 24
    header_h = 58
    rows_count = int(math.ceil(len(cards) / float(cols)))
    canvas_w = margin * 2 + cols * card_w + (cols - 1) * gap
    canvas_h = header_h + margin + rows_count * card_h + max(0, rows_count - 1) * gap + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (242, 242, 240))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((margin, 18), _truncate_text(title, 120), fill=(20, 20, 20, 255))
    draw.text(
        (margin, 36),
        (
            "Ordered by execution stage; checkerboard pixels are ignored "
            "fit evidence, not a coordinate crop."
        ),
        fill=(70, 70, 70, 255),
    )

    for i, card in enumerate(cards):
        col = i % cols
        row = i // cols
        x = margin + col * (card_w + gap)
        y = header_h + row * (card_h + gap)
        canvas.paste(card, (x, y))

    canvas.save(group_dir / "debug_contact_sheet.png")

def _merge_artifact_rows(debug_rows, viewer_index):
    merged = []
    by_idx = {}
    for row in debug_rows or []:
        try:
            idx = int(row.get("global_index", row.get("loop_index", -1)))
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        copied = dict(row)
        copied.setdefault("debug_only", True)
        by_idx[idx] = copied

    for row in viewer_index or []:
        try:
            idx = int(row.get("global_index", row.get("loop_index", -1)))
        except (TypeError, ValueError):
            idx = -1
        if idx < 0:
            merged.append(row)
            continue
        base = dict(by_idx.get(idx, {}))
        base.update(row)
        base["debug_only"] = False
        by_idx[idx] = base

    seen = set()
    for row in list(debug_rows or []) + list(viewer_index or []):
        try:
            idx = int(row.get("global_index", row.get("loop_index", -1)))
        except (TypeError, ValueError):
            continue
        if idx in seen or idx not in by_idx:
            continue
        merged.append(by_idx[idx])
        seen.add(idx)

    return merged

def _save_wall_artifact_folders(per_building_out, geojson_base, viewer_index, run_started_at=None, debug_rows=None):
    """
    Move generated debug artifacts into group-wise folders.
    The building root stays small: GLB, viewer_index.json, viewer_bundle.npz,
    and the artifact folder.
    """
    if not SAVE_WALL_ARTIFACT_FOLDERS:
        return

    out_dir = Path(per_building_out)
    artifact_root = out_dir / WALL_ARTIFACT_FOLDER_NAME
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    top_files = [p for p in out_dir.iterdir() if p.is_file()]
    moved_total = 0

    global_dir = artifact_root / "_global"
    global_names = {
        f"{geojson_base}__debug_facade_groups.json",
        f"{geojson_base}__debug_facade_groups_topdown.png",
        f"{geojson_base}__debug_facade_groups_unwrapped.png",
    }
    for p in top_files:
        if p.name in global_names:
            moved_total += int(_move_if_file(p, global_dir) is not None)

    projection_staging_dir = out_dir / str(globals().get(
        "TEMP_GLOBAL_WALL_GROUP_IMAGE_STAGING_FOLDER",
        "_tmp_wall_group_image_projections",
    ))
    if projection_staging_dir.is_dir():
        export_name = str(globals().get(
            "TEMP_GLOBAL_WALL_GROUP_IMAGE_EXPORT_FOLDER",
            "wall_group_image_projections",
        ))
        global_dir.mkdir(parents=True, exist_ok=True)
        projection_dst = global_dir / export_name
        if projection_dst.exists():
            if projection_dst.is_dir():
                shutil.rmtree(projection_dst)
            else:
                projection_dst.unlink()
        shutil.move(str(projection_staging_dir), str(projection_dst))
        moved_total += sum(1 for p in projection_dst.rglob("*") if p.is_file())

    artifact_rows = _merge_artifact_rows(debug_rows, viewer_index)

    grouped_rows = defaultdict(list)
    for row in artifact_rows:
        try:
            wall_idx = int(row.get("global_index", row.get("loop_index", -1)))
        except (TypeError, ValueError):
            wall_idx = -1
        if wall_idx < 0:
            continue

        group_tag = row.get("facade_group_tag")
        if group_tag:
            key = ("group", str(group_tag))
        else:
            key = ("wall", wall_idx)
        grouped_rows[key].append(row)

    group_dir_by_wall_index = {}
    contact_sheet_specs = []
    for key, rows in grouped_rows.items():
        wall_indices = []
        for row in rows:
            try:
                wall_indices.append(int(row.get("global_index", row.get("loop_index", -1))))
            except (TypeError, ValueError):
                pass
        wall_indices = sorted(set(i for i in wall_indices if i >= 0))
        wall_label = (
            f"walls_{wall_indices[0]:02d}-{wall_indices[-1]:02d}"
            if wall_indices else "walls_unknown"
        )

        if key[0] == "group":
            group_tag = key[1]
            group_id = rows[0].get("facade_group_id", "x")
            folder_name = (
                f"group_{int(group_id):02d}__{_safe_artifact_folder_part(group_tag)}__{wall_label}"
                if isinstance(group_id, int)
                else f"group_{_safe_artifact_folder_part(group_id)}__{_safe_artifact_folder_part(group_tag)}__{wall_label}"
            )
            prefixes = [
                f"{geojson_base}__{group_tag}__",
                f"sv__{geojson_base}__{group_tag}__",
            ]
            for row in rows:
                try:
                    row_wall_idx = int(row.get("global_index", row.get("loop_index", -1)))
                except (TypeError, ValueError):
                    row_wall_idx = -1
                if row_wall_idx >= 0:
                    prefixes.extend([
                        f"{geojson_base}_wall{row_wall_idx:02d}_",
                        f"wall_{row_wall_idx:02d}__",
                    ])
        else:
            wall_idx = key[1]
            wall_tag = rows[0].get("wall_tag", f"wall_{wall_idx:02d}")
            folder_name = f"single_wall_{wall_idx:02d}__{_safe_artifact_folder_part(wall_tag)}"
            prefixes = [
                f"{geojson_base}_wall{wall_idx:02d}_",
                f"wall_{wall_idx:02d}__",
            ]

        group_dir = artifact_root / folder_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for wall_idx in wall_indices:
            group_dir_by_wall_index[int(wall_idx)] = group_dir

        summary = {
            "artifact_unit": "facade_group" if key[0] == "group" else "single_wall",
            "folder": folder_name,
            "wall_indices": wall_indices,
            "rows": rows,
        }
        with open(group_dir / "group_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        explicit_names = set()
        for row in rows:
            for field in ("ortho_png", "sv_rgb_jpg"):
                value = row.get(field)
                if value:
                    explicit_names.add(Path(str(value)).name)

        moved_by_name = {}
        top_files = [p for p in out_dir.iterdir() if p.is_file()]

        for p in top_files:
            belongs = (
                any(p.name.startswith(prefix) for prefix in prefixes)
                or p.name in explicit_names
            )
            if belongs:
                moved = _move_if_file(p, group_dir)
                if moved is not None:
                    moved_total += 1
                    moved_by_name[p.name] = moved

        for row in rows:
            for field in ("ortho_png", "sv_rgb_jpg"):
                value = row.get(field)
                if not value:
                    continue
                moved = moved_by_name.get(Path(str(value)).name)
                if moved is not None:
                    row[field] = moved.relative_to(out_dir).as_posix()

        contact_sheet_specs.append((
            group_dir,
            folder_name,
            rows,
            summary["artifact_unit"],
        ))

    # A wall index is already encoded in every per-wall/group artifact name.
    # Use it as the final ownership fallback instead of inventing a catch-all
    # pseudo-group. This also recovers artifacts written before
    # a late processing exception can append its viewer-index row.
    keep_root_names = {"viewer_index.json", "viewer_bundle.npz"}
    for p in [x for x in out_dir.iterdir() if x.is_file()]:
        if (
            p.name in keep_root_names
            or p.suffix.lower()
            not in {".png", ".jpg", ".jpeg", ".json", ".npy"}
        ):
            continue
        wall_idx = _artifact_wall_index_from_name(p.name)
        destination = group_dir_by_wall_index.get(wall_idx)
        if destination is None:
            # Files with no wall identity are building-wide diagnostics.
            destination = global_dir
        moved_total += int(_move_if_file(p, destination) is not None)

    # Build sheets only after every artifact has reached its real owner. An
    # unresolved wall group therefore contains exactly its summary and the
    # single geometry/legend contact sheet.
    for group_dir, folder_name, rows, artifact_unit in contact_sheet_specs:
        _save_artifact_contact_sheet(
            group_dir,
            folder_name,
            rows,
            artifact_rows,
            artifact_unit=artifact_unit,
        )

    print(f"Saved group artifact folders: {artifact_root} ({len(grouped_rows)} folders, {moved_total} moved files)")

def _facade_selection_score(inside_ratio, wall_cover, outside_ratio, center_dist, area):
    return (
        4.0 * inside_ratio +
        3.0 * wall_cover -
        2.0 * outside_ratio -
        1.0 * center_dist +
        0.15 * np.log1p(area)
    )

def _select_facade_instances_for_wall(
    facade_stack,
    roof_mask,
    wall_poly_xy,
    H,
    W,
    wall_mask_override=None,
):
    if facade_stack.shape[0] == 0:
        return np.zeros((H, W), dtype=bool), [], []

    if roof_mask is None:
        roof_mask = np.zeros((H, W), dtype=bool)

    if wall_mask_override is None:
        wall_mask = _polygon_to_mask(H, W, wall_poly_xy)
    else:
        wall_mask = np.asarray(wall_mask_override, dtype=bool)
        if wall_mask.shape != (H, W):
            raise ValueError("wall_mask_override shape does not match the SAM image.")
    wall_area = max(int(wall_mask.sum()), 1)

    wy, wx = np.where(wall_mask)
    if len(wx) == 0:
        scored = []
        best_idx = -1
        best_score = -1e18
        best_mask = np.zeros((H, W), dtype=bool)
        for i in range(facade_stack.shape[0]):
            cand = facade_stack[i] & (~roof_mask)
            area = int(cand.sum())
            score = float(area)
            scored.append((i, score, area, 0, 0, 0.0))
            if score > best_score:
                best_score = score
                best_idx = i
                best_mask = cand
        return best_mask, ([best_idx] if best_idx >= 0 else []), scored

    wall_cx = float(wx.mean())
    wall_cy = float(wy.mean())
    diag = max(float(np.hypot(W, H)), 1.0)

    candidates = []
    scored = []

    for i in range(facade_stack.shape[0]):
        cand = facade_stack[i] & (~roof_mask)
        area = int(cand.sum())
        if area == 0:
            continue

        inter = int((cand & wall_mask).sum())
        outside = int((cand & (~wall_mask)).sum())

        cy, cx = np.where(cand)
        cand_cx = float(cx.mean())
        cand_cy = float(cy.mean())
        center_dist = np.hypot(cand_cx - wall_cx, cand_cy - wall_cy) / diag

        inside_ratio = inter / max(area, 1)
        wall_cover = inter / wall_area
        outside_ratio = outside / max(area, 1)

        score = _facade_selection_score(
            inside_ratio=inside_ratio,
            wall_cover=wall_cover,
            outside_ratio=outside_ratio,
            center_dist=center_dist,
            area=area,
        )

        scored.append((i, score, area, inter, outside, center_dist))
        candidates.append({
            "idx": int(i),
            "mask": cand,
            "score": float(score),
            "area": int(area),
            "inter": int(inter),
            "outside": int(outside),
            "center_dist": float(center_dist),
            "inside_ratio": float(inside_ratio),
            "wall_cover": float(wall_cover),
            "outside_ratio": float(outside_ratio),
        })

    if not candidates:
        return np.zeros((H, W), dtype=bool), [], scored

    ranked = sorted(candidates, key=lambda r: r["score"], reverse=True)
    selected = [ranked[0]]
    selected_mask = ranked[0]["mask"].copy()
    selected_inside = selected_mask & wall_mask

    if ENABLE_MULTI_FACADE_INSTANCE_SELECTION:
        max_selected = max(1, int(FACADE_INSTANCE_MAX_SELECTED))
        for cand_info in ranked[1:]:
            if len(selected) >= max_selected:
                break

            inter = max(int(cand_info["inter"]), 0)
            if inter <= 0:
                continue

            cand_inside = cand_info["mask"] & wall_mask
            new_inside = int((cand_inside & (~selected_inside)).sum())
            new_wall_cover = new_inside / wall_area
            duplicate_inside_ratio = 1.0 - (new_inside / max(inter, 1))

            if cand_info["inside_ratio"] < float(FACADE_INSTANCE_MIN_INSIDE_RATIO):
                continue
            if cand_info["outside_ratio"] > float(FACADE_INSTANCE_MAX_OUTSIDE_RATIO):
                continue
            if cand_info["center_dist"] > float(FACADE_INSTANCE_MAX_CENTER_DIST):
                continue
            if new_inside < int(FACADE_INSTANCE_MIN_NEW_INSIDE_PX):
                continue
            if new_wall_cover < float(FACADE_INSTANCE_MIN_NEW_WALL_COVER):
                continue
            if duplicate_inside_ratio > float(FACADE_INSTANCE_MAX_DUPLICATE_INSIDE_RATIO):
                continue

            selected.append(cand_info)
            selected_mask |= cand_info["mask"]
            selected_inside |= cand_inside

    selected_indices = [int(r["idx"]) for r in selected]
    return selected_mask, selected_indices, scored

def _gate_mask_stack(mask_stack, gate_mask):
    if mask_stack is None or mask_stack.shape[0] == 0:
        return np.zeros((0,) + gate_mask.shape, dtype=bool)

    gated = []
    for i in range(mask_stack.shape[0]):
        mask = np.asarray(mask_stack[i], dtype=bool) & gate_mask
        if mask.any():
            gated.append(mask)
    if not gated:
        return np.zeros((0,) + gate_mask.shape, dtype=bool)
    return np.stack(gated, axis=0)

def _select_facade_mask_with_optional_refinement(
    processor,
    state,
    facade_stack,
    roof_mask,
    wall_poly_xy,
    H,
    W,
    primary_prompt,
    refinement_prompt=None,
    wall_mask_override=None,
):
    primary_mask, primary_idxs, primary_scores = _select_facade_instances_for_wall(
        facade_stack=facade_stack,
        roof_mask=roof_mask,
        wall_poly_xy=wall_poly_xy,
        H=H,
        W=W,
        wall_mask_override=wall_mask_override,
    )
    primary_mask = np.asarray(primary_mask, dtype=bool)
    wall_mask = (
        _polygon_to_mask(H, W, wall_poly_xy)
        if wall_mask_override is None
        else np.asarray(wall_mask_override, dtype=bool)
    )

    info = {
        "enabled": bool(globals().get("ENABLE_FACADE_PROMPT_REFINEMENT", False)),
        "accepted": False,
        "reason": "disabled",
        "primary_prompt": str(primary_prompt),
        "refinement_prompt": str(refinement_prompt) if refinement_prompt else None,
        "primary_instances": int(facade_stack.shape[0]) if facade_stack is not None else 0,
        "primary_selected_indices": [int(i) for i in primary_idxs],
        "primary_pixels": int(primary_mask.sum()),
        "primary_wall_pixels": int((primary_mask & wall_mask).sum()),
    }

    if not info["enabled"]:
        return primary_mask, primary_idxs, primary_scores, facade_stack, info
    if not refinement_prompt:
        info["reason"] = "missing_refinement_prompt"
        return primary_mask, primary_idxs, primary_scores, facade_stack, info
    if str(refinement_prompt).strip().lower() == str(primary_prompt).strip().lower():
        info["reason"] = "same_prompt"
        return primary_mask, primary_idxs, primary_scores, facade_stack, info
    if info["primary_pixels"] <= 0:
        info["reason"] = "empty_primary"
        return primary_mask, primary_idxs, primary_scores, facade_stack, info

    out_refine = processor.set_text_prompt(state=state, prompt=refinement_prompt)
    refine_stack_raw = _extract_mask_stack(out_refine, H, W)
    refine_stack = _gate_mask_stack(refine_stack_raw, primary_mask)
    info["refinement_instances_raw"] = int(refine_stack_raw.shape[0])
    info["refinement_instances_gated"] = int(refine_stack.shape[0])
    if refine_stack.shape[0] == 0:
        info["reason"] = "no_refinement_instances_inside_primary"
        return primary_mask, primary_idxs, primary_scores, facade_stack, info

    refined_mask, refined_idxs, refined_scores = _select_facade_instances_for_wall(
        facade_stack=refine_stack,
        roof_mask=roof_mask,
        wall_poly_xy=wall_poly_xy,
        H=H,
        W=W,
        wall_mask_override=wall_mask_override,
    )
    refined_mask = np.asarray(refined_mask, dtype=bool) & primary_mask
    refined_pixels = int(refined_mask.sum())
    refined_wall_pixels = int((refined_mask & wall_mask).sum())
    primary_wall_pixels = max(int(info["primary_wall_pixels"]), 1)
    min_abs = max(1, int(globals().get("FACADE_REFINEMENT_MIN_WALL_PIXELS", 500)))
    min_ratio = float(globals().get("FACADE_REFINEMENT_MIN_PRIMARY_WALL_RATIO", 0.18))
    min_by_ratio = int(math.ceil(primary_wall_pixels * min_ratio))
    min_required = min(primary_wall_pixels, max(min_abs, min_by_ratio))
    info.update({
        "refinement_selected_indices": [int(i) for i in refined_idxs],
        "refinement_pixels": int(refined_pixels),
        "refinement_wall_pixels": int(refined_wall_pixels),
        "refinement_primary_wall_ratio": float(refined_wall_pixels / primary_wall_pixels),
        "refinement_min_required_wall_pixels": int(min_required),
    })

    if refined_pixels <= 0:
        info["reason"] = "empty_refinement_selection"
        return primary_mask, primary_idxs, primary_scores, facade_stack, info
    if refined_wall_pixels < min_required:
        info["reason"] = "insufficient_refinement_wall_support"
        return primary_mask, primary_idxs, primary_scores, facade_stack, info

    info["accepted"] = True
    info["reason"] = "accepted"
    return refined_mask, refined_idxs, refined_scores, refine_stack, info

def _select_best_facade_instance(facade_stack, roof_mask, wall_poly_xy, H, W):
    """Compatibility wrapper. The selector now returns one or more indices."""
    return _select_facade_instances_for_wall(facade_stack, roof_mask, wall_poly_xy, H, W)


def _depth_aware_region_fit_config():
    return DepthAwareRegionFitConfig(
        allow_rotation=bool(globals().get("DEPTH_AWARE_REGION_FIT_ALLOW_ROTATION", True)),
        max_working_dimension_px=int(globals().get("DEPTH_AWARE_REGION_FIT_MAX_WORKING_DIM_PX", 360)),
        max_translation_px=float(globals().get("DEPTH_AWARE_REGION_FIT_MAX_TRANSLATION_PX", 100.0)),
        scale_min=float(globals().get("DEPTH_AWARE_REGION_FIT_SCALE_MIN", 0.80)),
        scale_max=float(globals().get("DEPTH_AWARE_REGION_FIT_SCALE_MAX", 1.20)),
        max_rotation_deg=float(globals().get("DEPTH_AWARE_REGION_FIT_MAX_ROTATION_DEG", 5.0)),
        target_component_search_margin_px=float(globals().get("DEPTH_AWARE_REGION_FIT_SEARCH_MARGIN_PX", 120.0)),
        minimum_score_improvement=float(globals().get("DEPTH_AWARE_REGION_FIT_MIN_SCORE_IMPROVEMENT", 0.035)),
        minimum_iou_improvement=float(globals().get("DEPTH_AWARE_REGION_FIT_MIN_IOU_IMPROVEMENT", 0.025)),
        minimum_boundary_improvement=float(globals().get("DEPTH_AWARE_REGION_FIT_MIN_BOUNDARY_IMPROVEMENT", 0.035)),
        minimum_final_iou=float(globals().get("DEPTH_AWARE_REGION_FIT_MIN_FINAL_IOU", 0.30)),
        minimum_final_precision=float(globals().get("DEPTH_AWARE_REGION_FIT_MIN_FINAL_PRECISION", 0.55)),
    )

def _model_depth_boundary_fit_config(
    *,
    semantic_target_supported=True,
    background_aware=False,
):
    semantic_target_supported = bool(semantic_target_supported)
    background_aware = bool(background_aware)
    if background_aware and semantic_target_supported:
        scale_delta = float(globals().get(
            "MODEL_DEPTH_BACKGROUND_AWARE_MAX_SCALE_DELTA",
            0.25,
        ))
    else:
        scale_delta = float(globals().get(
            "MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_SCALE_DELTA"
            if semantic_target_supported
            else "MODEL_DEPTH_BOUNDARY_MICRO_MAX_SCALE_DELTA",
            0.10 if semantic_target_supported else 0.06,
        ))
    translation_norm = float(globals().get(
        "MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_TRANSLATION_PX"
        if semantic_target_supported
        else "MODEL_DEPTH_BOUNDARY_MICRO_MAX_TRANSLATION_PX",
        50.0 if semantic_target_supported else 20.0,
    ))
    displacement_limit = float(globals().get(
        "MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_MEAN_DISPLACEMENT_PX"
        if semantic_target_supported
        else "MODEL_DEPTH_BOUNDARY_MICRO_MAX_MEAN_DISPLACEMENT_PX",
        55.0 if semantic_target_supported else 20.0,
    ))
    coarse_translation_limit = min(48.0, translation_norm)
    return make_production_fit_config(
        allow_rotation=bool(globals().get("MODEL_DEPTH_BOUNDARY_FIT_ALLOW_ROTATION", False)),
        coarse_scale_min=1.0 - scale_delta,
        coarse_scale_max=1.0 + scale_delta,
        coarse_tx_min=-coarse_translation_limit,
        coarse_tx_max=coarse_translation_limit,
        coarse_tx_step=8.0 if semantic_target_supported else 4.0,
        coarse_ty_min=-coarse_translation_limit,
        coarse_ty_max=coarse_translation_limit,
        coarse_ty_step=8.0 if semantic_target_supported else 4.0,
        maximum_translation_x_px=translation_norm,
        maximum_translation_y_px=translation_norm,
        maximum_translation_norm_px=translation_norm,
        maximum_translation_norm_fraction=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_TRANSLATION_FRACTION",
            0.25,
        )),
        maximum_mean_displacement_px=displacement_limit,
        maximum_mean_displacement_fraction=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_ANCHOR_MAX_DISPLACEMENT_FRACTION",
            0.25,
        )),
        minimum_anchor_iou=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_ANCHOR_MIN_IOU",
            0.35,
        )),
        weight_translation_prior=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_ANCHOR_TRANSLATION_PRIOR_WEIGHT",
            0.40,
        )),
        translation_prior_sigma_x=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_ANCHOR_TRANSLATION_PRIOR_SIGMA_PX",
            40.0,
        )),
        translation_prior_sigma_y=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_ANCHOR_TRANSLATION_PRIOR_SIGMA_PX",
            40.0,
        )),
        image_border_epsilon_px=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_IMAGE_BORDER_EPSILON_PX",
            0.5,
        )),
        minimum_score_improvement=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_FIT_MIN_SCORE_IMPROVEMENT",
            0.025,
        )),
        weight_semantic_boundary=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_SEMANTIC_IMAGE_WEIGHT",
            1.35,
        )),
        semantic_boundary_sigma_px=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_SEMANTIC_IMAGE_SIGMA_PX",
            6.0,
        )),
        maximum_semantic_score_drop=float(globals().get(
            "MODEL_DEPTH_BOUNDARY_MAX_SEMANTIC_SCORE_DROP",
            0.03,
        )),
        minimum_masked_evidence_sample_count=int(globals().get(
            "MODEL_DEPTH_BOUNDARY_MIN_MASKED_EVIDENCE_SAMPLES",
            8,
        )),
    )


def _model_depth_prefit_semantic_config(
    image_shape_hw=None,
    *,
    target_wall_visibility=False,
):
    if image_shape_hw is None:
        scale_factor = 1.0
    else:
        scale_factor = float(np.clip(
            max(int(image_shape_hw[0]), int(image_shape_hw[1])) / 640.0,
            0.75,
            3.0,
        ))
    search_margin_name = (
        "MODEL_DEPTH_PREFIT_TARGET_WALL_SEARCH_MARGIN_PX"
        if target_wall_visibility
        else "MODEL_DEPTH_PREFIT_SEARCH_MARGIN_PX"
    )
    association_margin_name = (
        "MODEL_DEPTH_PREFIT_TARGET_WALL_ASSOCIATION_MARGIN_PX"
        if target_wall_visibility
        else "MODEL_DEPTH_PREFIT_ASSOCIATION_MARGIN_PX"
    )
    return PrefitSemanticGuidanceConfig(
        search_dilation_px=int(round(scale_factor * int(globals().get(
            search_margin_name,
            32 if target_wall_visibility else 96,
        )))),
        target_association_distance_px=scale_factor * float(globals().get(
            association_margin_name,
            24 if target_wall_visibility else 48,
        )),
        target_min_overlap_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_MIN_TARGET_PROJECTION_OVERLAP",
            0.01,
        )),
        target_max_instances_per_role=int(globals().get(
            "MODEL_DEPTH_PREFIT_MAX_TARGET_INSTANCES",
            4,
        )),
        minimum_instance_area_px=int(round(
            scale_factor * scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_MIN_INSTANCE_AREA_PX",
                80,
            ))
        )),
        # This is a literal image-space safety radius, not a resolution-scaled
        # fitting parameter.
        occluder_dilation_px=int(globals().get(
            "MODEL_DEPTH_PREFIT_OCCLUDER_DILATION_PX",
            3,
        )),
        generic_non_target_enabled=bool(globals().get(
            "ENABLE_MODEL_DEPTH_PREFIT_GENERIC_NON_TARGET",
            True,
        )),
        generic_non_target_min_target_coverage=float(globals().get(
            "MODEL_DEPTH_PREFIT_GENERIC_MIN_TARGET_COVERAGE",
            0.20,
        )),
        generic_non_target_projection_inset_px=int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_GENERIC_PROJECTION_INSET_PX",
                3,
            ))
        )),
        generic_non_target_target_dilation_px=int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_GENERIC_TARGET_DILATION_PX",
                2,
            ))
        )),
        generic_non_target_min_component_area_px=int(round(
            scale_factor * scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_GENERIC_MIN_COMPONENT_AREA_PX",
                80,
            ))
        )),
        generic_non_target_max_component_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_GENERIC_MAX_COMPONENT_FRACTION",
            0.20,
        )),
        generic_non_target_max_total_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_GENERIC_MAX_TOTAL_FRACTION",
            0.45,
        )),
        generic_non_target_max_target_overlap_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_GENERIC_MAX_TARGET_OVERLAP_FRACTION",
            0.15,
        )),
        context_adjacency_px=int(round(scale_factor * int(globals().get(
            "MODEL_DEPTH_PREFIT_INTERFACE_DILATION_PX",
            5,
        )))),
        boundary_thickness_px=int(round(scale_factor * int(globals().get(
            "MODEL_DEPTH_PREFIT_BOUNDARY_THICKNESS_PX",
            2,
        )))),
        image_border_exclusion_px=max(
            1,
            int(round(scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_IMAGE_BORDER_EXCLUSION_PX",
                2,
            )))),
        ),
        strict_roof_guidance_enabled=bool(globals().get(
            "ENABLE_MODEL_DEPTH_PREFIT_STRICT_ROOF_GUIDANCE",
            True,
        )),
        strict_roof_projected_band_radius_px=max(8, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_BAND_RADIUS_PX",
                18,
            ))
        ))),
        strict_roof_upper_building_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_STRICT_ROOF_UPPER_BUILDING_FRACTION",
            0.48,
        )),
        strict_roof_attachment_radius_px=max(4, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_ATTACHMENT_RADIUS_PX",
                8,
            ))
        ))),
        strict_roof_min_band_pixels=max(6, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_BAND_PIXELS",
                12,
            ))
        ))),
        strict_roof_min_band_span_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_BAND_SPAN_FRACTION",
            0.03,
        )),
        strict_roof_min_attachment_pixels=max(6, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_ATTACHMENT_PIXELS",
                12,
            ))
        ))),
        strict_roof_max_explicit_foreground_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_STRICT_ROOF_MAX_FOREGROUND_FRACTION",
            0.35,
        )),
        strict_roof_context_radius_px=max(2, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_CONTEXT_RADIUS_PX",
                3,
            ))
        ))),
        strict_roof_foreground_guard_radius_px=max(
            int(globals().get("MODEL_DEPTH_PREFIT_OCCLUDER_DILATION_PX", 3)),
            int(round(scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_FOREGROUND_GUARD_RADIUS_PX",
                4,
            )))),
        ),
        strict_roof_vegetation_projection_inset_px=max(1, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_VEGETATION_INSET_PX",
                2,
            ))
        ))),
        strict_roof_vegetation_inside_offset_px=max(4, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_VEGETATION_INSIDE_OFFSET_PX",
                8,
            ))
        ))),
        strict_roof_min_guide_component_pixels=max(3, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_MIN_GUIDE_COMPONENT_PIXELS",
                5,
            ))
        ))),
        strict_roof_bridge_enabled=bool(globals().get(
            "ENABLE_MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_DIAGNOSTIC",
            True,
        )),
        strict_roof_bridge_min_endpoint_run_px=max(2, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_MIN_ENDPOINT_RUN_PX",
                3,
            ))
        ))),
        strict_roof_bridge_max_gap_px=max(24, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_MAX_GAP_PX",
                64,
            ))
        ))),
        strict_roof_bridge_domain_dilation_px=max(1, int(round(
            scale_factor * int(globals().get(
                "MODEL_DEPTH_PREFIT_STRICT_ROOF_BRIDGE_DOMAIN_DILATION_PX",
                2,
            ))
        ))),
    )


def _model_depth_prefit_prompt_library():
    configured = globals().get(
        "MODEL_DEPTH_PREFIT_SEMANTIC_PROMPT_LIBRARY",
        {
            "building": ("building",),
            "roof": ("building roof",),
            "sky": ("sky",),
            "vegetation": ("tree",),
            "ground": ("ground",),
            "occluder": ("vehicle", "traffic sign"),
            "generic_occluder": (
                "foreground object",
                "object in front of building",
            ),
        },
    )
    if not isinstance(configured, dict):
        raise ValueError(
            "MODEL_DEPTH_PREFIT_SEMANTIC_PROMPT_LIBRARY must be a mapping."
        )
    normalized = {}
    for raw_role, raw_prompts in configured.items():
        if isinstance(raw_prompts, str):
            raw_prompts = (raw_prompts,)
        prompts = []
        for raw_prompt in raw_prompts or ():
            prompt = str(raw_prompt).strip()
            if prompt and prompt not in prompts:
                prompts.append(prompt)
        normalized[str(raw_role)] = prompts
    return normalized


def _model_depth_prefit_downstream_roof_prompts():
    """Return bare-roof prompts reserved for downstream cleanup evidence."""
    configured = globals().get(
        "MODEL_DEPTH_PREFIT_DOWNSTREAM_ROOF_PROMPTS",
        ("roof",),
    )
    if isinstance(configured, str):
        configured = (configured,)
    prompts = []
    for raw_prompt in configured or ():
        prompt = str(raw_prompt).strip()
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    if (
        bool(globals().get(
            "ENABLE_MODEL_DEPTH_PREFIT_STRICT_ROOF_GUIDANCE",
            True,
        ))
        and not prompts
    ):
        raise ValueError(
            "Strict pre-fit roof guidance requires at least one distinct "
            "MODEL_DEPTH_PREFIT_DOWNSTREAM_ROOF_PROMPTS entry."
        )
    return prompts


def _run_model_depth_prefit_semantic_guidance(
    *,
    processor,
    image_rgb,
    raw_projection_mask,
    target_wall_projection_mask=None,
    external_exclusion_mask=None,
    stage,
):
    """Run one automatic SAM3 prompt library and anchor it to the model mask."""
    if not bool(globals().get(
        "ENABLE_MODEL_DEPTH_PREFIT_SEMANTIC_GUIDANCE",
        True,
    )):
        return None

    model_image = image_rgb.convert("RGB")
    projection = np.asarray(raw_projection_mask, dtype=bool)
    expected_shape = (int(model_image.height), int(model_image.width))
    if projection.shape != expected_shape:
        raise ValueError(
            "Pre-fit semantic projection mask must match the PIL RGB image."
        )
    if target_wall_projection_mask is None:
        target_wall_projection = None
    else:
        target_wall_projection = np.asarray(
            target_wall_projection_mask,
            dtype=bool,
        )
        if target_wall_projection.shape != expected_shape:
            raise ValueError(
                "Target-wall semantic projection must match the PIL RGB image."
            )
    if external_exclusion_mask is None:
        external_exclusion = None
    else:
        external_exclusion = np.asarray(external_exclusion_mask, dtype=bool)
        if external_exclusion.shape != expected_shape:
            raise ValueError(
                "Pre-fit external exclusion must match the PIL RGB image."
            )

    prompt_library = _model_depth_prefit_prompt_library()
    downstream_roof_prompts = _model_depth_prefit_downstream_roof_prompts()
    role_stacks = {}
    prompt_results = []
    embedding_computed = False
    embedding_error = None
    state = None
    if processor is None:
        embedding_error = "sam3_processor_not_provided"
    else:
        try:
            # Always pass PIL RGB. The installed SAM3 processor reads NumPy HWC
            # dimensions incorrectly, and RGBA would leak hidden source pixels.
            with torch.no_grad():
                state = processor.set_image(model_image)
            embedding_computed = True
        except Exception as exc:
            embedding_error = (
                f"set_image_failed: {type(exc).__name__}: {exc}"
            )

    height, width = expected_shape
    for role, prompts in prompt_library.items():
        stacks = []
        for prompt in prompts:
            row = {
                "role": str(role),
                "prompt": str(prompt),
                "instance_count": 0,
                "status": "not_run",
            }
            if state is None:
                row["status"] = "image_embedding_unavailable"
                prompt_results.append(row)
                continue
            try:
                with torch.no_grad():
                    output = processor.set_text_prompt(
                        state=state,
                        prompt=prompt,
                    )
                stack = _extract_mask_stack(output, height, width)
                if stack.shape[0] > 0:
                    stacks.append(stack)
                row.update({
                    "status": "ok",
                    "instance_count": int(stack.shape[0]),
                    "pixel_count": int(stack.any(axis=0).sum())
                    if stack.shape[0] > 0 else 0,
                })
            except Exception as exc:
                row.update({
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            prompt_results.append(row)
        role_stacks[role] = (
            np.concatenate(stacks, axis=0)
            if stacks
            else np.zeros((0, height, width), dtype=bool)
        )

    downstream_roof_stacks = []
    for prompt in downstream_roof_prompts:
        row = {
            "role": "downstream_roof",
            "prompt": str(prompt),
            "instance_count": 0,
            "status": "not_run",
        }
        if state is None:
            row["status"] = "image_embedding_unavailable"
            prompt_results.append(row)
            continue
        try:
            with torch.no_grad():
                output = processor.set_text_prompt(
                    state=state,
                    prompt=prompt,
                )
            stack = _extract_mask_stack(output, height, width)
            if stack.shape[0] > 0:
                downstream_roof_stacks.append(stack)
            row.update({
                "status": "ok",
                "instance_count": int(stack.shape[0]),
                "pixel_count": int(stack.any(axis=0).sum())
                if stack.shape[0] > 0 else 0,
            })
        except Exception as exc:
            row.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        prompt_results.append(row)
    downstream_roof_stack = (
        np.concatenate(downstream_roof_stacks, axis=0)
        if downstream_roof_stacks
        else np.zeros((0, height, width), dtype=bool)
    )

    guidance = build_prefit_semantic_guidance(
        role_stacks,
        projection,
        config=_model_depth_prefit_semantic_config(projection.shape),
        external_exclusion_mask=external_exclusion,
        downstream_roof_mask_stack=downstream_roof_stack,
    )
    whole_model_target_pixels = int(np.asarray(
        guidance.get("target_semantic_mask", np.zeros(projection.shape, dtype=bool)),
        dtype=bool,
    ).sum())
    whole_model_projection_pixels = int(projection.sum())
    whole_model_target_area_ratio = float(
        whole_model_target_pixels / max(whole_model_projection_pixels, 1)
    )
    guidance["whole_model_projection_pixels"] = whole_model_projection_pixels
    target_wall_guidance = None
    if target_wall_projection is not None:
        target_wall_guidance = build_prefit_semantic_guidance(
            role_stacks,
            target_wall_projection,
            config=_model_depth_prefit_semantic_config(
                projection.shape,
                target_wall_visibility=True,
            ),
            external_exclusion_mask=external_exclusion,
            downstream_roof_mask_stack=downstream_roof_stack,
        )
    semantic_instance_count = int(sum(
        stack.shape[0] for stack in role_stacks.values()
    ))
    runtime_metadata = {
        "enabled": True,
        "stage": str(stage),
        "method": (
            "sam3_prompt_library_plus_projection_local_non_target_residual"
        ),
        "manual_prompt_required": False,
        "promptless_everything_mode": False,
        "prompt_library": {
            role: list(prompts)
            for role, prompts in prompt_library.items()
        },
        "downstream_roof_prompts": list(downstream_roof_prompts),
        "prompt_results": prompt_results,
        "image_embedding_computed": bool(embedding_computed),
        "image_embedding_error": embedding_error,
        "semantic_instance_count": semantic_instance_count,
        "downstream_roof_instance_count": int(
            downstream_roof_stack.shape[0]
        ),
        "roof_prompt_stage_separated": True,
        "segmentation_available": bool(semantic_instance_count > 0),
        # Even a failed SAM call retains the hard projection-local search mask.
        "used_for_fitting": True,
    }
    metadata = dict(guidance.get("metadata", {}))
    metadata.update(runtime_metadata)
    guidance["metadata"] = metadata
    guidance["raw_projection_mask"] = projection.copy()
    if target_wall_guidance is not None:
        target_metadata = dict(target_wall_guidance.get("metadata", {}))
        target_metadata.update(runtime_metadata)
        target_metadata.update({
            "used_for_fitting": False,
            "used_for_candidate_visibility": True,
            "projection_scope": "target_wall",
            "whole_model_target_semantic_pixels": whole_model_target_pixels,
            "whole_model_projection_pixels": whole_model_projection_pixels,
            "whole_model_target_area_ratio": whole_model_target_area_ratio,
        })
        target_wall_guidance["metadata"] = target_metadata
        target_wall_guidance["whole_model_target_area_ratio"] = (
            whole_model_target_area_ratio
        )
        target_wall_guidance["whole_model_projection_pixels"] = (
            whole_model_projection_pixels
        )
        guidance["target_wall_guidance"] = target_wall_guidance
    return guidance


def _combine_model_depth_fit_evidence(
    guidance,
    image_shape_hw,
    *,
    external_exclusion_mask=None,
    background_aware=False,
):
    """Combine SAM/locality and OSM masks without changing canvas coordinates."""
    height, width = (int(image_shape_hw[0]), int(image_shape_hw[1]))
    shape = (height, width)
    valid = None
    semantic_valid = None
    boundary_maps = None
    metadata = {}
    overlay_guidance = guidance

    if guidance is not None:
        valid_key = (
            "background_aware_valid_evidence_mask"
            if background_aware
            else "valid_evidence_mask"
        )
        boundary_key = (
            "background_aware_boundary_maps"
            if background_aware
            else "boundary_maps"
        )
        valid = np.asarray(guidance[valid_key], dtype=bool).copy()
        if valid.shape != shape:
            raise ValueError(
                "Pre-fit semantic evidence mask must match the fitting image."
            )
        boundary_maps = {
            str(label): np.asarray(mask, dtype=bool)
            for label, mask in dict(guidance.get(boundary_key, {})).items()
        }
        if background_aware:
            semantic_valid = np.asarray(
                guidance.get("semantic_valid_evidence_mask", valid),
                dtype=bool,
            ).copy()
            if semantic_valid.shape != shape:
                raise ValueError(
                    "Semantic evidence mask must match the fitting image."
                )
        metadata = dict(guidance.get("metadata", {}))
        metadata["used_for_fitting"] = True
        metadata["fit_evidence_mode"] = (
            "foreground_background_split"
            if background_aware
            else "legacy_incumbent"
        )
        if background_aware:
            metadata["legacy_boundary_pixels_by_class"] = dict(
                metadata.get("boundary_pixels_by_class", {})
            )
            metadata["boundary_pixels_by_class"] = {
                str(label): int(np.asarray(mask, dtype=bool).sum())
                for label, mask in boundary_maps.items()
            }

    external = None
    if external_exclusion_mask is not None:
        external = np.asarray(external_exclusion_mask, dtype=bool)
        if external.shape != shape:
            raise ValueError(
                "External-building exclusion must match semantic evidence."
            )
        if valid is None:
            valid = np.ones(shape, dtype=bool)
        valid &= ~external
        if semantic_valid is not None:
            semantic_valid &= ~external

    if valid is not None:
        metadata.update({
            "combined_valid_evidence_pixels": int(valid.sum()),
            "combined_excluded_evidence_pixels": int(valid.size - valid.sum()),
            "external_exclusion_pixels": int(external.sum())
            if external is not None else 0,
            "canvas_preserved": True,
        })
        if guidance is not None:
            overlay_guidance = dict(guidance)
            overlay_guidance["valid_evidence_mask"] = valid
            overlay_guidance["boundary_maps"] = boundary_maps
            if semantic_valid is not None:
                overlay_guidance["semantic_valid_evidence_mask"] = (
                    semantic_valid
                )
            if background_aware:
                overlay_guidance["background_aware_active"] = True
            local_search = np.asarray(
                guidance.get("local_search_mask", np.ones(shape, dtype=bool)),
                dtype=bool,
            )
            overlay_guidance["excluded_evidence_mask"] = (
                (~valid) & local_search
            )
            overlay_guidance["metadata"] = metadata

    return {
        "valid_evidence_mask": valid,
        "semantic_valid_evidence_mask": semantic_valid,
        "boundary_maps": boundary_maps,
        "metadata": metadata,
        "overlay_guidance": overlay_guidance,
    }


def _background_aware_recovery_eligibility(guidance):
    """Detect the general failure mode where foreground masking hides the roof.

    The threshold is intentionally semantic and wall-independent: recovery is
    considered only when more than half of the highest-priority roof guide was
    removed by the incumbent mask and the split restores a material run.
    """
    result = {
        "eligible": False,
        "reason": "guidance_unavailable",
        "legacy_roof_pixels": 0,
        "full_roof_pixels": 0,
        "restored_roof_pixels": 0,
        "legacy_roof_retention": 1.0,
    }
    if not isinstance(guidance, Mapping):
        return result
    legacy_maps = guidance.get("boundary_maps", {})
    full_maps = guidance.get("background_aware_boundary_maps", {})
    if not isinstance(legacy_maps, Mapping) or not isinstance(full_maps, Mapping):
        result["reason"] = "background_aware_maps_unavailable"
        return result
    legacy_roof = np.asarray(legacy_maps.get("roof", []), dtype=bool)
    full_roof = np.asarray(full_maps.get("roof", []), dtype=bool)
    if legacy_roof.ndim != 2 or full_roof.shape != legacy_roof.shape:
        result["reason"] = "roof_map_shape_mismatch"
        return result
    full_pixels = int(full_roof.sum())
    legacy_pixels = int((legacy_roof & full_roof).sum())
    restored_pixels = int((full_roof & (~legacy_roof)).sum())
    retention = float(legacy_pixels / max(full_pixels, 1))
    result.update({
        "legacy_roof_pixels": legacy_pixels,
        "full_roof_pixels": full_pixels,
        "restored_roof_pixels": restored_pixels,
        "legacy_roof_retention": retention,
    })
    if full_pixels <= 0:
        result["reason"] = "no_roof_guide"
        return result
    maximum_retention = float(globals().get(
        "MODEL_DEPTH_BACKGROUND_AWARE_MAX_LEGACY_ROOF_RETENTION",
        0.50,
    ))
    minimum_restored = int(globals().get(
        "MODEL_DEPTH_BACKGROUND_AWARE_MIN_RESTORED_ROOF_PIXELS",
        64,
    ))
    if retention > maximum_retention:
        result["reason"] = "legacy_roof_guide_sufficient"
        return result
    if restored_pixels < minimum_restored:
        result["reason"] = "restored_roof_run_too_small"
        return result
    result["eligible"] = True
    result["reason"] = "majority_of_roof_guide_was_masked"
    return result


def _common_semantic_alignment(
    fit_result,
    boundary_maps,
    image_shape_hw,
    fit_config,
    *,
    included_classes=None,
):
    if not isinstance(fit_result, Mapping):
        return {
            "score": 0.0,
            "sample_count": 0,
            "classes": list(included_classes or []),
        }
    points = np.asarray(
        fit_result.get("fit_fitted_points", []),
        dtype=np.float64,
    ).reshape(-1, 2)
    segments = list(fit_result.get("fit_segment_indices", []))
    classes = list(fit_result.get("fit_segment_classes", []))
    weights = np.asarray(
        fit_result.get("fit_segment_weights", []),
        dtype=np.float64,
    ).reshape(-1)
    if (
        len(points) < 2
        or not segments
        or len(segments) != len(classes)
        or len(segments) != len(weights)
    ):
        return {
            "score": 0.0,
            "sample_count": 0,
            "classes": list(included_classes or []),
        }
    return semantic_boundary_alignment_score(
        points,
        segments,
        classes,
        weights,
        boundary_maps,
        image_shape_hw,
        config=fit_config,
        included_classes=included_classes,
    )


def _full_roof_alignment(fit_result, boundary_maps, image_shape_hw, fit_config):
    return _common_semantic_alignment(
        fit_result,
        boundary_maps,
        image_shape_hw,
        fit_config,
        included_classes=("roof",),
    )


def _choose_background_aware_fit(
    incumbent,
    challenger,
    *,
    full_boundary_maps,
    image_shape_hw,
    comparison_config,
    eligibility,
):
    """Choose a challenger only for a clear, anchor-safe roof improvement."""
    incumbent_roof = _full_roof_alignment(
        incumbent,
        full_boundary_maps,
        image_shape_hw,
        comparison_config,
    )
    challenger_roof = _full_roof_alignment(
        challenger,
        full_boundary_maps,
        image_shape_hw,
        comparison_config,
    )
    incumbent_common = _common_semantic_alignment(
        incumbent,
        full_boundary_maps,
        image_shape_hw,
        comparison_config,
    )
    challenger_common = _common_semantic_alignment(
        challenger,
        full_boundary_maps,
        image_shape_hw,
        comparison_config,
    )
    gain = float(challenger_roof["score"] - incumbent_roof["score"])
    minimum_gain = float(globals().get(
        "MODEL_DEPTH_BACKGROUND_AWARE_MIN_FULL_ROOF_SCORE_GAIN",
        0.08,
    ))
    common_semantic_gain = float(
        challenger_common["score"] - incumbent_common["score"]
    )
    maximum_common_drop = float(globals().get(
        "MODEL_DEPTH_BACKGROUND_AWARE_MAX_COMMON_SEMANTIC_SCORE_DROP",
        0.02,
    ))
    accepted = bool(
        challenger.get("applied", False)
        and challenger.get("anchor_motion_gate_passed", False)
        and challenger.get("anchor_iou_gate_passed", False)
        and int(challenger_roof.get("sample_count", 0)) > 0
        and gain >= minimum_gain
        and common_semantic_gain >= -maximum_common_drop
    )
    decision = {
        **dict(eligibility),
        "accepted": accepted,
        "reason": (
            "accepted_material_full_roof_alignment_gain"
            if accepted
            else (
                "kept_incumbent_common_semantic_regression"
                if common_semantic_gain < -maximum_common_drop
                else "kept_incumbent_no_material_full_roof_gain"
            )
        ),
        "incumbent_full_roof_score": float(incumbent_roof["score"]),
        "challenger_full_roof_score": float(challenger_roof["score"]),
        "full_roof_score_gain": gain,
        "minimum_full_roof_score_gain": minimum_gain,
        "incumbent_full_roof_samples": int(incumbent_roof["sample_count"]),
        "challenger_full_roof_samples": int(challenger_roof["sample_count"]),
        "incumbent_common_semantic_score": float(incumbent_common["score"]),
        "challenger_common_semantic_score": float(challenger_common["score"]),
        "common_semantic_score_gain": common_semantic_gain,
        "maximum_common_semantic_score_drop": maximum_common_drop,
        "incumbent_common_semantic_samples": int(
            incumbent_common["sample_count"]
        ),
        "challenger_common_semantic_samples": int(
            challenger_common["sample_count"]
        ),
    }
    if not accepted:
        # Returning the original object preserves every incumbent field and
        # transform exactly when the new interpretation is not demonstrably
        # better.
        return incumbent, decision
    selected = dict(challenger)
    selected["background_aware_recovery"] = decision
    return selected, decision


def _finalize_selected_osm_masked_depth_refit(
    fit_result,
    *,
    raw_wall_outline_px,
    exclusion_mask,
    valid_evidence_mask=None,
):
    """Make the clean selected-source refit authoritative over preselection."""
    result = dict(fit_result)
    exclusion = np.asarray(exclusion_mask, dtype=bool)
    if exclusion.ndim != 2:
        raise ValueError("Selected OSM refit exclusion must be a 2D mask.")
    if valid_evidence_mask is None:
        combined_valid_evidence = ~exclusion
    else:
        supplied_valid_evidence = np.asarray(valid_evidence_mask, dtype=bool)
        valid_evidence = supplied_valid_evidence
        if valid_evidence.shape != exclusion.shape:
            raise ValueError(
                "Selected OSM refit evidence mask must match its exclusion mask."
            )
        combined_valid_evidence = valid_evidence & (~exclusion)
    combined_exclusion = ~combined_valid_evidence
    numerical_fit_applied = bool(result.get("applied", False))
    identity_diagnostics = dict(result.get("diagnostics_before", {}))
    identity_evidence_sample_count = int(
        identity_diagnostics.get("evidence_sample_count", 0)
    )
    result.update({
        "selected_source_osm_refit": True,
        "selected_source_osm_refit_numerical_fit_applied": numerical_fit_applied,
        "selected_source_osm_refit_identity_fallback": False,
        "osm_excluded_image_evidence_pixel_count": int(exclusion.sum()),
        "semantic_or_locality_excluded_image_evidence_pixel_count": int(
            (combined_exclusion & (~exclusion)).sum()
        ),
        "valid_image_evidence_pixel_count": int(
            combined_valid_evidence.sum()
        ),
        "excluded_image_evidence_pixel_count": int(combined_exclusion.sum()),
        "excluded_image_evidence_column_count": int(
            combined_exclusion.any(axis=0).sum()
        ),
    })
    if numerical_fit_applied:
        return result

    numerical_transform = dict(result.get("transform", {}))
    numerical_semantic_score_after = float(
        result.get("semantic_boundary_score_after", 0.0)
    )
    identity_transform = {
        **identity_diagnostics,
        "scale": 1.0,
        "rotation_deg": 0.0,
        "tx": 0.0,
        "ty": 0.0,
        "score": float(result.get("score_before", 0.0)),
        "transform_center_x": float(
            numerical_transform.get("transform_center_x", 0.0)
        ),
        "transform_center_y": float(
            numerical_transform.get("transform_center_y", 0.0)
        ),
        "reference_visible_length": float(
            numerical_transform.get("reference_visible_length", 0.0)
        ),
    }
    result.update({
        "selected_source_osm_refit_numerical_reason": str(
            result.get("reason", "unknown")
        ),
        "selected_source_osm_refit_numerical_transform": numerical_transform,
        "selected_source_osm_refit_numerical_score_improvement": float(
            result.get("score_improvement", 0.0)
        ),
        "selected_source_osm_refit_numerical_semantic_boundary_score_after": (
            numerical_semantic_score_after
        ),
        "selected_source_osm_refit_identity_fallback": True,
        "applied": True,
        "reason": (
            "selected_osm_refit_kept_unshifted_raw_projection_no_evidence"
            if identity_evidence_sample_count <= 0
            else "selected_osm_refit_kept_unshifted_raw_projection"
        ),
        "homography": np.eye(3, dtype=np.float64),
        "fitted_points": np.asarray(
            result.get("original_points", []),
            dtype=np.float64,
        ).copy(),
        "depth_global_fitted_wall_outline_px": np.asarray(
            raw_wall_outline_px,
            dtype=np.float64,
        ).copy(),
        "score_after": float(result.get("score_before", 0.0)),
        "score_improvement": 0.0,
        "mean_vertex_displacement_px": 0.0,
        "semantic_boundary_score_after": float(
            result.get("semantic_boundary_score_before", 0.0)
        ),
        "transform": identity_transform,
    })
    return result


def _prepare_osm_building_occlusion_context(
    geojson_path,
    base_z,
    *,
    camera_elevation_resolver=None,
):
    configured = bool(globals().get(
        "ENABLE_OSM_EXTERNAL_BUILDING_OCCLUSION",
        True,
    ))
    depth_global_active = bool(
        str(globals().get("FACADE_ALIGNMENT_MODE", "depth_global")).strip().lower()
        == "depth_global"
        and globals().get("ENABLE_MODEL_DEPTH_BOUNDARY_FIT", False)
    )
    enabled = bool(configured and depth_global_active)
    context = {
        "enabled": enabled,
        "available": False,
        "reason": (
            "disabled"
            if not configured
            else "requires_depth_global_alignment"
            if not depth_global_active
            else "not_initialized"
        ),
        "blocker_meshes": [],
        "blocker_lookup": {},
        "metadata": None,
        "excluded_target_buildings": [],
        "blocker_terrain_metadata": {},
        "blocker_terrain_source": "not_available",
    }
    if not enabled:
        return context

    try:
        model_geometry = build_model_occlusion_geometry(geojson_path)
        cache_path = Path(str(globals().get(
            "OSM_BUILDING_CACHE_DIR",
            "cache/osm_building_occlusion",
        )))
        if not cache_path.is_absolute():
            cache_path = Path(__file__).resolve().parents[1] / cache_path
        buildings, osm_metadata = fetch_osm_buildings(
            model_footprint=model_geometry["footprint"],
            source_crs=str(globals().get("SOURCE_CRS", "EPSG:25832")),
            radius_m=float(globals().get(
                "OSM_BUILDING_QUERY_RADIUS_M",
                120.0,
            )),
            endpoint=str(globals().get(
                "OSM_BUILDING_OVERPASS_ENDPOINT",
                DEFAULT_OVERPASS_ENDPOINT,
            )),
            cache_dir=cache_path,
            refresh=bool(globals().get(
                "OSM_BUILDING_REFRESH_CACHE",
                False,
            )),
            timeout_s=float(globals().get(
                "OSM_BUILDING_OVERPASS_TIMEOUT_S",
                90.0,
            )),
            default_height_m=float(globals().get(
                "OSM_BUILDING_DEFAULT_HEIGHT_M",
                15.0,
            )),
            level_height_m=float(globals().get(
                "OSM_BUILDING_LEVEL_HEIGHT_M",
                3.0,
            )),
        )
        blockers, excluded = remove_target_osm_buildings(
            buildings,
            model_geometry["footprint"],
        )
        terrain_sampler = None
        terrain_reason = "disabled"
        if bool(globals().get("OSM_BUILDING_USE_DGM_TERRAIN", True)):
            validation = getattr(camera_elevation_resolver, "validation", None)
            sampler = getattr(camera_elevation_resolver, "sampler", None)
            if bool(getattr(validation, "consistent", False)) and sampler is not None:
                terrain_sampler = sampler.sample
                terrain_reason = "validated_dgm"
            else:
                terrain_reason = (
                    "dgm_base_validation_unavailable_or_failed"
                )
        blocker_terrain_metadata = {}
        blocker_meshes, blocker_lookup = build_osm_blocker_meshes(
            blockers,
            ground_z=float(base_z),
            ground_z_sampler=terrain_sampler,
            maximum_terrain_samples=int(globals().get(
                "OSM_BUILDING_TERRAIN_MAX_SAMPLES",
                9,
            )),
            terrain_top_margin_m=float(globals().get(
                "OSM_BUILDING_TERRAIN_TOP_MARGIN_M",
                0.5,
            )),
            terrain_metadata=blocker_terrain_metadata,
        )
        context.update({
            "available": True,
            "reason": "available",
            "blocker_meshes": blocker_meshes,
            "blocker_lookup": blocker_lookup,
            "metadata": osm_metadata,
            "excluded_target_buildings": excluded,
            "blocker_terrain_metadata": blocker_terrain_metadata,
            "blocker_terrain_source": terrain_reason,
            "nearby_building_count": int(len(buildings)),
            "external_blocker_count": int(len(blocker_meshes)),
        })
        print(
            "OSM external-building occlusion: "
            f"{len(blocker_meshes)} blocker(s) from {len(buildings)} nearby building(s)."
        )
    except Exception as exc:
        context.update({
            "available": False,
            "reason": f"osm_context_failed: {type(exc).__name__}: {exc}",
        })
        print(
            "OSM external-building occlusion unavailable; continuing with "
            f"the existing source criteria ({exc})."
        )
    return context


def _score_candidate_external_building_occlusion(
    source,
    *,
    target_meshes,
    target_quads,
    osm_context,
    fit_H,
    facade_tag,
    source_index,
):
    if not bool(osm_context.get("available", False)):
        return None
    if not target_meshes or not target_quads:
        source["external_building_occlusion_reason"] = "target_geometry_unavailable"
        return None

    try:
        raw_target_depth = source.get("_external_building_raw_target_depth")
        if raw_target_depth is None:
            raw_target_depth = render_model_depth_map(
                target_meshes,
                source["K"],
                source["Rwc"],
                source["C"],
                source["img"].size,
                near_m=float(globals().get("MODEL_DEPTH_NEAR_M", 0.05)),
            )
            source["_external_building_raw_target_depth"] = raw_target_depth
        width, height = source["img"].size
        candidate = {
            "camera_utm_xyz": [
                float(value) for value in np.asarray(source["C"]).tolist()
            ],
            "projection_heading_deg": float(source.get(
                "projection_heading", source.get("heading", 0.0),
            )),
            "heading_deg": float(source.get("heading", 0.0)),
            "pitch_deg": float(source.get("pitch", 0.0)),
            "fov_deg": float(source.get("fov", 100.0)),
        }
        occlusion = evaluate_candidate_occlusion(
            candidate=candidate,
            target_meshes=target_meshes,
            target_quads=target_quads,
            blocker_meshes=osm_context.get("blocker_meshes", []),
            blocker_lookup=osm_context.get("blocker_lookup", {}),
            image_size=f"{width}x{height}",
            depth_tolerance_m=float(globals().get(
                "OSM_BUILDING_DEPTH_TOLERANCE_M", 0.10,
            )),
            corridor_buffer_m=float(globals().get(
                "OSM_BUILDING_CORRIDOR_BUFFER_M", 1.0,
            )),
            require_corridor_intersection=bool(globals().get(
                "OSM_BUILDING_REQUIRE_CORRIDOR_INTERSECTION",
                False,
            )),
            target_alignment_H=fit_H,
            precomputed_raw_target_depth=raw_target_depth,
        )
        fraction = float(occlusion["osm_occluded_fraction"])
        clear_threshold = float(globals().get(
            "OSM_BUILDING_CLEAR_OCCLUSION_FRACTION", 0.005,
        ))
        source.update({
            "external_building_occlusion_available": True,
            "external_building_occlusion_evaluation_failed": False,
            "external_building_occlusion_fraction": fraction,
            "external_building_clear": bool(fraction <= clear_threshold),
            "external_building_occlusion_mask": np.asarray(
                occlusion["occlusion_mask"], dtype=bool,
            ),
            "external_building_target_mask": np.asarray(
                occlusion["target_mask"], dtype=bool,
            ),
            # Keep the complete rendered neighbouring-building footprint for
            # diagnostics.  Candidate preselection refines this into
            # ``external_building_fit_exclusion_mask`` by retaining pixels
            # outside the model plus blockers physically in front of it.
            "external_building_blocker_mask": np.isfinite(
                np.asarray(occlusion["blocker_depth"], dtype=np.float32)
            ),
            "external_building_candidate_blockers": list(
                occlusion["candidate_blocker_mesh_names"]
            ),
            "external_building_candidate_blocker_terrain": {
                name: dict(osm_context.get(
                    "blocker_terrain_metadata",
                    {},
                ).get(name, {}))
                for name in occlusion["candidate_blocker_mesh_names"]
            },
            "external_building_blocker_terrain_source": str(
                osm_context.get("blocker_terrain_source", "not_available")
            ),
            "external_building_occlusion_reason": (
                "clear" if fraction <= clear_threshold else "obstructed"
            ),
        })
        source["_external_building_blocker_depth"] = np.asarray(
            occlusion["blocker_depth"],
            dtype=np.float32,
        )
        return occlusion
    except Exception as exc:
        source["external_building_occlusion_evaluation_failed"] = True
        source["external_building_occlusion_reason"] = (
            f"osm_candidate_scoring_failed: {type(exc).__name__}: {exc}"
        )
        print(
            f"[{facade_tag}] candidate {source_index:02d} OSM scoring failed; "
            f"continuing without an OSM exclusion ({exc})."
        )
        return None


def _assess_target_wall_candidate_visibility(target_wall_guidance):
    if target_wall_guidance is None:
        return {
            "accepted": True,
            "fallback_used": True,
            "reason": "target_wall_semantic_guidance_unavailable",
            "segmentation_available": False,
        }
    return assess_prefit_candidate_visibility(
        target_wall_guidance,
        minimum_target_projection_pixels=int(globals().get(
            "MODEL_DEPTH_PREFIT_VISIBILITY_MIN_TARGET_PIXELS",
            250,
        )),
        minimum_target_support_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_VISIBILITY_MIN_TARGET_SUPPORT_FRACTION",
            0.10,
        )),
        maximum_occluder_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_VISIBILITY_MAX_OCCLUDER_FRACTION",
            0.80,
        )),
        low_support_occluder_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_VISIBILITY_LOW_SUPPORT_OCCLUDER_FRACTION",
            0.60,
        )),
        minimum_largest_visible_component_fraction=float(globals().get(
            "MODEL_DEPTH_PREFIT_VISIBILITY_MIN_LARGEST_COMPONENT_FRACTION",
            0.05,
        )),
        maximum_whole_model_target_area_ratio=float(globals().get(
            "MODEL_DEPTH_PREFIT_VISIBILITY_MAX_TARGET_AREA_RATIO",
            6.0,
        )),
        reject_when_target_semantics_absent=bool(globals().get(
            "MODEL_DEPTH_PREFIT_VISIBILITY_REJECT_NO_TARGET",
            True,
        )),
    )


def _osm_prefit_hard_rejection(source):
    """Return a rejection reason only for a nearly fully OSM-hidden wall."""
    if not bool(source.get("external_building_occlusion_available", False)):
        return None
    fraction = float(source.get("external_building_occlusion_fraction", 0.0))
    threshold = float(globals().get(
        "OSM_BUILDING_PREFIT_HARD_REJECT_FRACTION",
        0.97,
    ))
    if fraction < threshold:
        return None
    return (
        "osm_nearly_fully_blocks_raw_target_projection: "
        f"{fraction:.6f} >= {threshold:.6f}"
    )


def _candidate_depth_global_and_osm_preselection(
    sources,
    *,
    processor=None,
    outline_xyz,
    target_meshes,
    target_quads,
    meshes_named,
    model_boundary_edges_xyz,
    osm_context,
    facade_tag,
):
    """OSM-gate, semantically assess, then anchor-fit every candidate."""
    for source_index, source in enumerate(sources):
        source["selection_projection_H"] = np.eye(3, dtype=np.float64)
        source["depth_global_fit_evaluated_before_selection"] = True
        source["depth_global_fit_applied"] = False
        source["depth_global_fit_reason"] = "not_run"
        source["depth_global_score_improvement"] = 0.0
        source["depth_global_candidate_usable"] = True
        source["depth_global_candidate_rejection_reason"] = None
        source["depth_global_target_visibility"] = None
        source["depth_global_sam3_skipped"] = False
        source["depth_global_sam3_skip_reason"] = None
        source["external_building_occlusion_available"] = False
        source["external_building_occlusion_evaluation_failed"] = False
        source["external_building_clear"] = False
        source["external_building_occlusion_reason"] = str(
            osm_context.get("reason", "not_available")
        )
        source["external_building_candidate_blockers"] = []
        source["external_building_candidate_blocker_terrain"] = {}
        source["external_building_blocker_terrain_source"] = str(
            osm_context.get("blocker_terrain_source", "not_available")
        )

        try:
            (
                near_clipped_points,
                near_clipped_segments,
                _near_clipped_world_points,
                projection_info,
            ) = project_outline_world_edges_near_clipped(
                np.asarray(outline_xyz, dtype=np.float64),
                source["K"],
                source["Rwc"],
                source["C"],
                near_m=FACADE_PROJECTION_NEAR_PLANE_M,
            )
            source["wireframe_projection_info"] = projection_info
            full_raw_outline, _ = project_points_world_to_image(
                np.asarray(outline_xyz, dtype=np.float64),
                source["K"],
                source["Rwc"],
                source["C"],
                clip_behind=False,
            )
            uses_near_clipped_projection = not bool(
                projection_info["full_outline_topology_valid"]
            )
            visible_clipped_outline = project_polygon_world_to_image_clipped(
                np.asarray(outline_xyz, dtype=np.float64),
                source["K"],
                source["Rwc"],
                source["C"],
                source["img"].size,
                near_m=FACADE_PROJECTION_NEAR_PLANE_M,
                clip_to_image=True,
            )
            raw_outline = (
                visible_clipped_outline
                if uses_near_clipped_projection
                else full_raw_outline
            )
            visible_projection_valid = bool(
                raw_outline.ndim == 2
                and raw_outline.shape[0] >= 3
                and raw_outline.shape[1] == 2
                and np.isfinite(raw_outline).all()
                and not _closed_polyline_self_intersects(raw_outline)
            )
            source.update({
                "selection_uses_near_clipped_projection": (
                    uses_near_clipped_projection
                ),
                "selection_projection_topology_valid": visible_projection_valid,
                "selection_visible_wall_outline_px": raw_outline,
                "selection_full_wall_outline_px": full_raw_outline,
                "selection_real_wall_edge_points_px": near_clipped_points,
                "selection_real_wall_edge_segments": [
                    (int(index0), int(index1))
                    for index0, index1 in near_clipped_segments
                ],
                "depth_global_raw_wall_outline_px": raw_outline,
            })
            if not visible_projection_valid:
                reason = "no_visible_wall_polygon_after_near_plane_clipping"
                source.update({
                    "depth_global_fit_result": {
                        "homography": np.eye(3, dtype=np.float64),
                        "applied": False,
                        "reason": reason,
                    },
                    "depth_global_fit_applied": False,
                    "depth_global_fit_reason": reason,
                    "depth_global_corrected_wall_outline_px": raw_outline,
                    "depth_global_candidate_usable": False,
                    "depth_global_candidate_rejection_reason": reason,
                    "external_building_occlusion_reason": (
                        "skipped_no_visible_wall_projection"
                    ),
                })
                print(
                    f"[{facade_tag}] candidate {source_index:02d} has no visible "
                    "wall polygon after near-plane clipping"
                )
                continue

            full_depth = _render_model_depth_view(
                meshes_named=meshes_named,
                K=source["K"],
                R_wc=source["Rwc"],
                C=source["C"],
                source_image_size=source["img"].size,
            )
            if full_depth is None or not np.any(
                np.isfinite(full_depth) & (full_depth > 0.0)
            ):
                raise ValueError("whole model is not visible in the candidate")

            full_projection_mask = (
                np.isfinite(full_depth) & (full_depth > 0.0)
            )
            target_wall_projection_mask = None
            if target_meshes:
                raw_target_depth = render_model_depth_map(
                    target_meshes,
                    source["K"],
                    source["Rwc"],
                    source["C"],
                    source["img"].size,
                    near_m=float(globals().get("MODEL_DEPTH_NEAR_M", 0.05)),
                )
                target_wall_projection_mask = (
                    np.isfinite(raw_target_depth)
                    & (raw_target_depth > 0.0)
                    & full_projection_mask
                    & (
                        np.abs(raw_target_depth - full_depth)
                        <= float(globals().get(
                            "FACADE_SOURCE_VISIBILITY_DEPTH_TOLERANCE_M",
                            0.05,
                        ))
                    )
                )
                source["depth_global_target_wall_projection_mask"] = (
                    target_wall_projection_mask.copy()
                )
                source["_external_building_raw_target_depth"] = np.where(
                    target_wall_projection_mask,
                    raw_target_depth,
                    np.nan,
                ).astype(np.float32)

            # OSM must be evaluated in the raw canvas before SAM association or
            # fitting. The former ordering let the broad "building" prompt and
            # generic image edges lock onto the very neighbour later removed.
            raw_osm = _score_candidate_external_building_occlusion(
                source,
                target_meshes=target_meshes,
                target_quads=target_quads,
                osm_context=osm_context,
                fit_H=np.eye(3, dtype=np.float64),
                facade_tag=facade_tag,
                source_index=source_index,
            )
            raw_osm_failure = str(source.get(
                "external_building_occlusion_reason",
                "",
            ))
            if (
                raw_osm is None
                and bool(osm_context.get("available", False))
                and bool(source.get(
                    "external_building_occlusion_evaluation_failed",
                    False,
                ))
            ):
                # Once nearby OSM geometry is available, a per-candidate
                # render/scoring error is not evidence that the view is clear.
                # Keeping the old raw-ranking fallback could silently select
                # exactly the unverified foreground-building case that this
                # stage is meant to prevent.
                reason = f"rejected_before_sam3_{raw_osm_failure}"
                source.update({
                    "depth_global_fit_result": {
                        "homography": np.eye(3, dtype=np.float64),
                        "applied": False,
                        "reason": reason,
                    },
                    "depth_global_full_model_depth": full_depth,
                    "depth_global_fit_applied": False,
                    "depth_global_fit_reason": reason,
                    "depth_global_corrected_wall_outline_px": raw_outline,
                    "depth_global_candidate_usable": False,
                    "depth_global_candidate_rejection_reason": reason,
                    "depth_global_sam3_skipped": True,
                    "depth_global_sam3_skip_reason": raw_osm_failure,
                })
                print(
                    f"[{facade_tag}] candidate {source_index:02d} rejected "
                    "before SAM3 because OSM scoring failed"
                )
                continue
            if raw_osm is not None:
                source.update({
                    "external_building_raw_projection_occlusion_fraction": float(
                        source.get("external_building_occlusion_fraction", 0.0)
                    ),
                    "external_building_raw_projection_occlusion_mask": np.asarray(
                        source.get(
                            "external_building_occlusion_mask",
                            np.zeros(full_depth.shape, dtype=bool),
                        ),
                        dtype=bool,
                    ).copy(),
                    "external_building_raw_projection_target_mask": np.asarray(
                        source.get(
                            "external_building_target_mask",
                            np.zeros(full_depth.shape, dtype=bool),
                        ),
                        dtype=bool,
                    ).copy(),
                })
            blocker_depth = source.get("_external_building_blocker_depth")
            osm_blocker_mask = None
            if blocker_depth is not None:
                blocker_depth = np.asarray(blocker_depth, dtype=np.float32)
                depth_tolerance = float(globals().get(
                    "OSM_BUILDING_DEPTH_TOLERANCE_M",
                    0.10,
                ))
                # Outside the projected model, ignore every visible neighbour;
                # inside it, exclude only blockers physically in front. This
                # preserves valid target pixels when an OSM footprint is behind
                # the target along the same image ray.
                osm_blocker_mask = np.isfinite(blocker_depth) & (
                    (~full_projection_mask)
                    | (blocker_depth + depth_tolerance < full_depth)
                )
                source["external_building_fit_exclusion_mask"] = (
                    osm_blocker_mask.copy()
                )

            hard_osm_rejection = _osm_prefit_hard_rejection(source)
            if hard_osm_rejection is not None:
                reason = f"skipped_before_sam3_{hard_osm_rejection}"
                source.update({
                    "depth_global_fit_result": {
                        "homography": np.eye(3, dtype=np.float64),
                        "applied": False,
                        "reason": reason,
                    },
                    "depth_global_full_model_depth": full_depth,
                    "depth_global_fit_applied": False,
                    "depth_global_fit_reason": reason,
                    "depth_global_corrected_wall_outline_px": raw_outline,
                    "depth_global_candidate_usable": False,
                    "depth_global_candidate_rejection_reason": reason,
                    "depth_global_sam3_skipped": True,
                    "depth_global_sam3_skip_reason": hard_osm_rejection,
                })
                print(
                    f"[{facade_tag}] candidate {source_index:02d} rejected before "
                    f"SAM3 ({hard_osm_rejection})"
                )
                continue

            prefit_semantic_guidance = None
            try:
                prefit_semantic_guidance = (
                    _run_model_depth_prefit_semantic_guidance(
                        processor=processor,
                        image_rgb=source["img"],
                        raw_projection_mask=full_projection_mask,
                        target_wall_projection_mask=target_wall_projection_mask,
                        external_exclusion_mask=osm_blocker_mask,
                        stage="candidate_preselection_before_global_depth_fit",
                    )
                )
            except Exception as exc:
                source["depth_global_prefit_semantic_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                print(
                    f"[{facade_tag}] candidate {source_index:02d} pre-fit "
                    f"semantic guidance unavailable ({exc}); using geometric fit."
                )
            source["depth_global_prefit_semantic_guidance"] = (
                prefit_semantic_guidance
            )
            source["depth_global_prefit_semantic_metadata"] = (
                dict(prefit_semantic_guidance.get("metadata", {}))
                if prefit_semantic_guidance is not None else None
            )
            source["depth_global_fit_semantic_guidance"] = (
                prefit_semantic_guidance
            )
            target_wall_guidance = (
                prefit_semantic_guidance.get("target_wall_guidance")
                if prefit_semantic_guidance is not None else None
            )
            target_visibility = _assess_target_wall_candidate_visibility(
                target_wall_guidance
            )
            source["depth_global_target_visibility"] = target_visibility
            semantic_fit_anchor_supported = bool(
                target_visibility.get("segmentation_available", False)
                and float(target_visibility.get(
                    "target_support_fraction",
                    0.0,
                )) >= float(globals().get(
                    "MODEL_DEPTH_PREFIT_VISIBILITY_MIN_TARGET_SUPPORT_FRACTION",
                    0.10,
                ))
            )
            source["depth_global_semantic_fit_anchor_supported"] = (
                semantic_fit_anchor_supported
            )
            source["depth_global_candidate_usable"] = bool(
                target_visibility.get("accepted", True)
            )
            if not source["depth_global_candidate_usable"]:
                reason = (
                    "semantic_visibility_rejected_before_fit: "
                    f"{target_visibility.get('reason', 'unknown')}"
                )
                source.update({
                    "depth_global_fit_result": {
                        "homography": np.eye(3, dtype=np.float64),
                        "applied": False,
                        "reason": reason,
                    },
                    "depth_global_full_model_depth": full_depth,
                    "depth_global_fit_applied": False,
                    "depth_global_fit_reason": reason,
                    "depth_global_corrected_wall_outline_px": raw_outline,
                    "depth_global_candidate_rejection_reason": reason,
                })
                print(
                    f"[{facade_tag}] candidate {source_index:02d} rejected by "
                    f"target-wall visibility ({target_visibility['reason']})"
                )
                continue
            prefit_fit_evidence = _combine_model_depth_fit_evidence(
                prefit_semantic_guidance,
                full_depth.shape,
                external_exclusion_mask=osm_blocker_mask,
            )
            source["depth_global_fit_semantic_guidance"] = (
                prefit_fit_evidence["overlay_guidance"]
            )

            semantic_boundary_geometry = None
            if bool(globals().get(
                "MODEL_DEPTH_BOUNDARY_USE_SEMANTIC_GUIDES",
                True,
            )):
                try:
                    semantic_boundary_geometry = project_semantic_model_boundary_edges(
                        model_edges_xyz_by_class=model_boundary_edges_xyz or {},
                        K=source["K"],
                        R_wc=source["Rwc"],
                        C=source["C"],
                        full_model_depth=full_depth,
                        image_to_output_H=np.eye(3, dtype=np.float64),
                        near_m=float(globals().get("MODEL_DEPTH_NEAR_M", 0.05)),
                        sample_step_px=float(globals().get(
                            "MODEL_DEPTH_BOUNDARY_SEMANTIC_SAMPLE_STEP_PX", 2.0,
                        )),
                        silhouette_tolerance_px=float(globals().get(
                            "MODEL_DEPTH_BOUNDARY_SEMANTIC_SILHOUETTE_TOLERANCE_PX", 4.0,
                        )),
                        depth_search_radius_px=int(globals().get(
                            "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_SEARCH_RADIUS_PX", 2,
                        )),
                        depth_tolerance_m=float(globals().get(
                            "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_TOLERANCE_M", 0.35,
                        )),
                        depth_relative_tolerance=float(globals().get(
                            "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_RELATIVE_TOLERANCE", 0.03,
                        )),
                        maximum_visibility_gap_samples=int(globals().get(
                            "MODEL_DEPTH_BOUNDARY_SEMANTIC_MAX_GAP_SAMPLES", 2,
                        )),
                        minimum_visible_run_px=float(globals().get(
                            "MODEL_DEPTH_BOUNDARY_SEMANTIC_MIN_RUN_PX", 8.0,
                        )),
                    )
                except Exception as exc:
                    source["depth_global_semantic_guide_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            fit_image_bgr = cv2.cvtColor(
                np.asarray(source["img"].convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
            incumbent_fit_config = _model_depth_boundary_fit_config(
                semantic_target_supported=semantic_fit_anchor_supported,
            )
            common_fit_kwargs = dict(
                image_bgr=fit_image_bgr,
                full_model_depth=full_depth,
                raw_wall_outline_px=raw_outline,
                wall_local_fit_outline_px=raw_outline,
                minimum_area_px=int(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_MIN_AREA_PX", 350,
                )),
                minimum_component_fraction=float(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_MIN_COMPONENT_FRACTION", 0.02,
                )),
                contour_epsilon_px=float(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_CONTOUR_EPSILON_PX", 1.5,
                )),
                maximum_points=int(globals().get(
                    "MODEL_DEPTH_BOUNDARY_FIT_MAX_POINTS", 240,
                )),
                semantic_boundary_geometry=semantic_boundary_geometry,
                semantic_class_weights={
                    "roof": float(globals().get(
                        "MODEL_DEPTH_BOUNDARY_ROOF_WEIGHT", 3.0,
                    )),
                    "wall": float(globals().get(
                        "MODEL_DEPTH_BOUNDARY_WALL_WEIGHT", 2.0,
                    )),
                    "base": float(globals().get(
                        "MODEL_DEPTH_BOUNDARY_BASE_WEIGHT", 0.35,
                    )),
                },
            )
            fit_result = fit_depth_silhouette_to_image(
                **common_fit_kwargs,
                fit_config=incumbent_fit_config,
                valid_image_evidence_mask=prefit_fit_evidence[
                    "valid_evidence_mask"
                ],
                semantic_image_boundary_maps=prefit_fit_evidence[
                    "boundary_maps"
                ],
                semantic_image_guidance_metadata=prefit_fit_evidence[
                    "metadata"
                ],
            )
            recovery_eligibility = (
                _background_aware_recovery_eligibility(
                    prefit_semantic_guidance
                )
            )
            if (
                semantic_fit_anchor_supported
                and recovery_eligibility["eligible"]
            ):
                background_fit_evidence = _combine_model_depth_fit_evidence(
                    prefit_semantic_guidance,
                    full_depth.shape,
                    external_exclusion_mask=osm_blocker_mask,
                    background_aware=True,
                )
                background_fit_config = _model_depth_boundary_fit_config(
                    semantic_target_supported=True,
                    background_aware=True,
                )
                background_fit_result = fit_depth_silhouette_to_image(
                    **common_fit_kwargs,
                    fit_config=background_fit_config,
                    valid_image_evidence_mask=background_fit_evidence[
                        "valid_evidence_mask"
                    ],
                    semantic_valid_image_evidence_mask=(
                        background_fit_evidence[
                            "semantic_valid_evidence_mask"
                        ]
                    ),
                    semantic_image_boundary_maps=background_fit_evidence[
                        "boundary_maps"
                    ],
                    semantic_image_guidance_metadata=(
                        background_fit_evidence["metadata"]
                    ),
                )
                fit_result, recovery_decision = (
                    _choose_background_aware_fit(
                        fit_result,
                        background_fit_result,
                        full_boundary_maps=background_fit_evidence[
                            "boundary_maps"
                        ],
                        image_shape_hw=full_depth.shape,
                        comparison_config=incumbent_fit_config,
                        eligibility=recovery_eligibility,
                    )
                )
                if recovery_decision["accepted"]:
                    prefit_fit_evidence = background_fit_evidence
                    fit_semantic_guidance = background_fit_evidence[
                        "overlay_guidance"
                    ]
                    source["depth_global_fit_semantic_guidance"] = (
                        fit_semantic_guidance
                    )
                    source["depth_global_fit_semantic_metadata"] = dict(
                        fit_semantic_guidance.get("metadata", {})
                    )
            fit_applied = bool(fit_result.get("applied", False))
            fit_H = np.asarray(
                fit_result.get("homography", np.eye(3)),
                dtype=np.float64,
            )
            if not fit_applied:
                fit_H = np.eye(3, dtype=np.float64)
            source.update({
                "selection_projection_H": fit_H,
                "depth_global_fit_result": fit_result,
                "depth_global_full_model_depth": full_depth,
                "depth_global_fit_applied": fit_applied,
                "depth_global_fit_reason": str(fit_result.get("reason", "unknown")),
                "depth_global_score_improvement": float(
                    fit_result.get("score_improvement", 0.0)
                ),
                "depth_global_corrected_wall_outline_px": apply_H(raw_outline, fit_H),
            })

            _score_candidate_external_building_occlusion(
                source,
                target_meshes=target_meshes,
                target_quads=target_quads,
                osm_context=osm_context,
                fit_H=fit_H,
                facade_tag=facade_tag,
                source_index=source_index,
            )

            fitted_osm_rejection = _osm_prefit_hard_rejection(source)
            if fitted_osm_rejection is not None:
                source["depth_global_candidate_usable"] = False
                source["depth_global_candidate_rejection_reason"] = (
                    "postfit_" + fitted_osm_rejection
                )

            status = "accepted" if fit_applied else "raw fallback"
            osm_text = (
                f", OSM blocked={100.0 * float(source.get('external_building_occlusion_fraction', 0.0)):.2f}%"
                if source.get("external_building_occlusion_available", False)
                else ""
            )
            print(
                f"[{facade_tag}] candidate {source_index:02d} depth-global {status} | "
                f"gain={source['depth_global_score_improvement']:.4f}{osm_text}"
            )
        except Exception as exc:
            source["depth_global_fit_reason"] = (
                f"candidate_preselection_failed: {type(exc).__name__}: {exc}"
            )
            source["external_building_occlusion_reason"] = (
                source["depth_global_fit_reason"]
            )
            print(
                f"[{facade_tag}] candidate {source_index:02d} global preselection "
                f"failed; keeping raw ranking ({exc})."
            )


def _model_boundary_edges_xyz_by_class(edge_groups, corners, id_to_idx):
    """Resolve labeled GeoJSON edges into world-space endpoint pairs."""
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
            endpoint_keys = [tuple(np.round(point, 7)) for point in edge]
            key = tuple(sorted(endpoint_keys))
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
        resolved[edge_class] = np.asarray(edges, dtype=np.float64).reshape(-1, 2, 3)
    return resolved

def _apply_facade_texture_to_fragments(group_records, texture_img, frame, S_m_to_px, out_w, out_h, mesh_by_name):
    textured = {}
    for rec in group_records:
        mesh = mesh_by_name.get(rec["mesh_name"])
        if mesh is None:
            continue
        uv_px = apply_H(frame["to_uv"](rec["wall_quad"]), S_m_to_px)
        uv = np.empty_like(uv_px, dtype=np.float64)
        uv[:, 0] = uv_px[:, 0] / float(out_w)
        uv[:, 1] = 1.0 - (uv_px[:, 1] / float(out_h))
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture_img)
        textured[int(rec["global_index"])] = uv_px
    return textured

def _resolve_facade_source_selection_policy(facade_group_items, configured_mode=None):
    mode = str(
        configured_mode
        if configured_mode is not None
        else globals().get("FACADE_SOURCE_SELECTION_MODE", "auto")
    ).strip().lower()
    valid_modes = {"auto", "projected_coverage", "legacy_wall_prism"}
    if mode not in valid_modes:
        raise ValueError(
            "FACADE_SOURCE_SELECTION_MODE must be 'auto', "
            "'projected_coverage', or 'legacy_wall_prism'."
        )

    nonempty_items = [item for item in facade_group_items if item.get("records")]
    singleton_only = bool(nonempty_items) and all(
        len(item["records"]) == 1 for item in nonempty_items
    )
    resolved = mode
    if mode == "auto":
        resolved = "legacy_wall_prism" if singleton_only else "projected_coverage"
    return resolved, singleton_only

def _texture_facade_group(group_records,
                          group_id,
                          geojson_base,
                          per_building_out,
                          base_z,
                          pano_records,
                          processor,
                          mesh_by_name,
                          meshes_named=None,
                          model_boundary_edges_xyz=None,
                          osm_occlusion_context=None,
                          stage_timer=None,
                          source_selection_policy="projected_coverage",
                          camera_elevation_resolver=None,
                          loop_records=None):
    with _timer_stage(stage_timer, f"facade group {group_id:02d} / geometry"):
        geom = _facade_group_geometry(group_records)
    if geom is None:
        return {}

    frame = geom["frame"]
    outline_xyz = geom["outline_xyz"]
    outline_m = geom["outline_m"]
    rect_m = geom["rect_m"]
    rect_xyz = geom["rect_xyz"]

    cid = group_records[0]["component_id"]
    lid = group_records[0]["loop_id"]
    cid_tag = _artifact_topology_id(cid)
    lid_tag = _artifact_topology_id(lid)
    first_idx = int(group_records[0]["global_index"])
    last_idx = int(group_records[-1]["global_index"])
    facade_tag = (
        f"c{cid_tag}_l{lid_tag}_g{group_id:02d}_"
        f"w{first_idx:02d}-{last_idx:02d}"
    )

    source_mode = "best_single_native_source_by_target_model_visibility"
    requested_alignment_mode = str(globals().get(
        "FACADE_ALIGNMENT_MODE", "depth_global",
    )).strip().lower()
    source_candidate_meta = []
    sv_jpg_name = None
    sv_jpg_path = None

    with _timer_stage(stage_timer, f"{facade_tag} / select pano candidates"):
        group_wall_quads = [rec["wall_quad"] for rec in group_records]
        group_wall_normals = [rec["normal"] for rec in group_records]
        pano_candidates = select_panos_for_facade_group(
            geom,
            pano_records,
            max_panos=int(FACADE_GROUP_MAX_CANDIDATE_PANOS),
            wall_quads=group_wall_quads,
            wall_normals=group_wall_normals,
        )
        if (
            bool(globals().get("FACADE_GROUP_RECOVERY_ENABLED", True))
            and source_selection_policy != "legacy_wall_prism"
            and facade_group_candidates_need_recovery(pano_candidates)
        ):
            recovered_records = discover_recovery_panos_for_facade_group(
                geom,
                transformer,
                back_tx,
                API_KEY,
                existing_records=pano_records,
            )
            if recovered_records:
                known_ids = {str(rec.get("pano_id", "")) for rec in pano_records}
                added = [
                    rec for rec in recovered_records
                    if str(rec.get("pano_id", "")) not in known_ids
                ]
                print(
                    f"[{facade_tag}] recovered {len(added)} outward-facing "
                    "Street View panorama candidate(s)."
                )
                pano_candidates = select_panos_for_facade_group(
                    geom,
                    list(pano_records) + added,
                    max_panos=int(FACADE_GROUP_MAX_CANDIDATE_PANOS),
                    wall_quads=group_wall_quads,
                    wall_normals=group_wall_normals,
                )
    with _timer_stage(stage_timer, f"{facade_tag} / fetch SV + source selection"):
        target_meshes = [
            (str(group_rec["mesh_name"]), mesh_by_name[str(group_rec["mesh_name"])])
            for group_rec in group_records
            if group_rec.get("mesh_name") is not None
            and str(group_rec["mesh_name"]) in mesh_by_name
        ]
        target_quads = [
            np.asarray(group_rec["wall_quad"], dtype=np.float64)
            for group_rec in group_records
            if np.asarray(group_rec.get("wall_quad"), dtype=np.float64).shape == (4, 3)
            and np.isfinite(np.asarray(group_rec["wall_quad"], dtype=np.float64)).all()
        ]
        candidate_preselection_evaluator = None
        if (
            requested_alignment_mode == "depth_global"
            and bool(globals().get("ENABLE_MODEL_DEPTH_BOUNDARY_FIT", False))
        ):
            candidate_preselection_evaluator = lambda sources: (
                _candidate_depth_global_and_osm_preselection(
                    sources,
                    processor=processor,
                    outline_xyz=outline_xyz,
                    target_meshes=target_meshes,
                    target_quads=target_quads,
                    meshes_named=meshes_named or [],
                    model_boundary_edges_xyz=model_boundary_edges_xyz or {},
                    osm_context=osm_occlusion_context or {
                        "enabled": False,
                        "available": False,
                        "reason": "building_osm_context_not_provided",
                        "blocker_meshes": [],
                        "blocker_lookup": {},
                    },
                    facade_tag=facade_tag,
                )
            )
        source_result = select_facade_source_from_panos(
            geom,
            pano_candidates,
            base_z,
            rect_xyz,
            outline_xyz,
            facade_tag=facade_tag,
            img_size=SV_SIZE,
            source_selection_policy=source_selection_policy,
            meshes_named=meshes_named or [],
            target_mesh_names=[
                str(group_rec["mesh_name"])
                for group_rec in group_records
                if group_rec.get("mesh_name") is not None
            ],
            facade_alignment_mode=requested_alignment_mode,
            candidate_preselection_evaluator=candidate_preselection_evaluator,
            camera_elevation_resolver=camera_elevation_resolver,
        )
    if source_result is None:
        print(f"[facade_g{group_id:02d}] Street View source preparation failed - grouped wall not textured.")
        return {}

    source_mode = str(source_result.get("source_mode", source_mode))
    rec = source_result["rec"]
    cam = np.asarray(source_result["camera_xyz"], dtype=np.float64)
    camera_elevation_info = dict(
        source_result.get("camera_elevation") or {}
    )
    heading = float(source_result["heading"])
    projection_heading = float(source_result.get("projection_heading", heading))
    meridian_convergence = float(
        source_result.get(
            "meridian_convergence_deg",
            wrap_delta_deg(heading, projection_heading),
        )
    )
    pitch = float(source_result["pitch"])
    fov_deg = float(source_result["fov"])
    img_rgb = source_result["image"].convert("RGB")
    selected_sources = list(source_result.get("sources", []))
    selected_source_index = int(source_result.get("selected_source_index", 0))
    selected_source = (
        selected_sources[selected_source_index]
        if 0 <= selected_source_index < len(selected_sources)
        else None
    )
    uv_rect = np.asarray(source_result["uv_rect"], dtype=np.float64)
    uv_outline = np.asarray(source_result["uv_outline"], dtype=np.float64)
    wall_only_uv_rect = uv_rect.copy()
    wall_only_uv_outline = uv_outline.copy()
    raw_uv_rect = np.asarray(
        source_result.get("uv_rect_before_wireframe_fit", wall_only_uv_rect),
        dtype=np.float64,
    )
    raw_uv_outline = np.asarray(
        source_result.get("uv_outline_before_wireframe_fit", wall_only_uv_outline),
        dtype=np.float64,
    )
    external_building_occlusion_info = dict(
        source_result.get("external_building_occlusion") or {}
    )
    selected_external_building_mask = source_result.get(
        "selected_external_building_removal_mask"
    )
    selected_external_building_target_mask = source_result.get(
        "selected_external_building_target_mask"
    )
    selected_external_building_local_mask = None
    selected_external_building_side_crop_info = {}
    if selected_external_building_mask is not None:
        selected_external_building_mask = np.asarray(
            selected_external_building_mask,
            dtype=bool,
        )
        expected_mask_shape = (int(img_rgb.height), int(img_rgb.width))
        if selected_external_building_mask.shape != expected_mask_shape:
            print(
                f"[{facade_tag}] external-building mask shape "
                f"{selected_external_building_mask.shape} does not match "
                f"the processing image {expected_mask_shape}; ignoring it."
            )
            external_building_occlusion_info.update({
                "fallback_mask_required": False,
                "mask_applied": False,
                "mask_reason": "shape_mismatch",
            })
            selected_external_building_mask = None
        elif not selected_external_building_mask.any():
            external_building_occlusion_info.update({
                "fallback_mask_required": False,
                "mask_applied": False,
                "mask_reason": "empty_mask",
            })
            selected_external_building_mask = None
        else:
            selected_external_building_local_mask = (
                selected_external_building_mask.copy()
            )
            if selected_external_building_target_mask is not None:
                selected_external_building_target_mask = np.asarray(
                    selected_external_building_target_mask,
                    dtype=bool,
                )
                if selected_external_building_target_mask.shape != expected_mask_shape:
                    selected_external_building_target_mask = None
            if selected_external_building_target_mask is None:
                candidate_fit = source_result.get(
                    "selected_candidate_depth_global_fit"
                )
                candidate_wall_outline = (
                    candidate_fit.get("depth_global_fitted_wall_outline_px")
                    if isinstance(candidate_fit, dict) else None
                )
                if candidate_wall_outline is not None:
                    selected_external_building_target_mask = _polygon_to_mask(
                        expected_mask_shape[0],
                        expected_mask_shape[1],
                        np.asarray(candidate_wall_outline, dtype=np.float64),
                    )
                else:
                    selected_external_building_target_mask = _polygon_to_mask(
                        expected_mask_shape[0],
                        expected_mask_shape[1],
                        uv_outline,
                    )
            (
                selected_external_building_mask,
                selected_external_building_side_crop_info,
            ) = _external_building_lr_side_exclusion_mask(
                selected_external_building_local_mask,
                selected_external_building_target_mask,
            )

    external_removal_mask_path = None
    external_removed_preview_path = None
    if selected_external_building_mask is not None:
        external_building_occlusion_info.update({
            "mask_applied": True,
            "mask_reason": "selected_source_lr_style_osm_side_exclusion",
            "raw_occlusion_pixel_count": int(
                selected_external_building_local_mask.sum()
            ),
            "removed_source_pixel_count": int(selected_external_building_mask.sum()),
            "lr_style_side_exclusion_pixel_count": int(
                selected_external_building_mask.sum()
            ),
            "lr_style_side_crop": selected_external_building_side_crop_info,
            "selected_source_depth_global_refit_required": True,
        })
        if (
            bool(globals().get("SAVE_OSM_BUILDING_OCCLUSION_DEBUG", True))
            or SAVE_ARTIFACT_CONTACT_SHEET
        ):
            external_removal_mask_path = Path(
                per_building_out,
                f"{geojson_base}__{facade_tag}__selected_external_building_removal_mask.png",
            )
            Image.fromarray(
                selected_external_building_mask.astype(np.uint8) * 255,
                mode="L",
            ).save(external_removal_mask_path)
            external_removed_preview_path = Path(
                per_building_out,
                f"{geojson_base}__{facade_tag}__selected_source_external_buildings_removed.png",
            )
            _save_external_building_removal_preview(
                img_rgb,
                selected_external_building_mask,
                external_removed_preview_path,
            )
    else:
        external_building_occlusion_info.setdefault("mask_applied", False)
        external_building_occlusion_info.setdefault(
            "selected_source_depth_global_refit_required",
            False,
        )
    facade_alignment_selection = None
    facade_alignment_info = None
    image_space_wireframe_fit = source_result.get("wireframe_fit")
    urls_fetched = list(source_result.get("urls_fetched", []))
    depth_artifacts = {}
    depth_fit_context = None
    depth_boundary_fit_result = None
    preselected_depth_boundary_fit_result = source_result.get(
        "selected_candidate_depth_global_fit"
    )
    preselected_full_model_depth = source_result.get(
        "selected_candidate_full_model_depth"
    )
    selected_full_model_depth_for_sides = None
    if preselected_full_model_depth is not None:
        candidate_side_depth = np.asarray(
            preselected_full_model_depth, dtype=np.float32
        )
        selected_raw_shape = (
            (int(selected_source["img"].height), int(selected_source["img"].width))
            if selected_source is not None
            else (int(img_rgb.height), int(img_rgb.width))
        )
        if candidate_side_depth.shape == selected_raw_shape:
            selected_full_model_depth_for_sides = candidate_side_depth.copy()
    depth_boundary_fit_info = {
        "enabled": bool(
            requested_alignment_mode == "depth_global"
            and globals().get("ENABLE_MODEL_DEPTH_BOUNDARY_FIT", False)
        ),
        "applied": False,
        "reason": "not_run",
        "uses_segmentation": False,
        "downstream_authoritative": False,
    }
    depth_boundary_artifacts = {}
    boundary_meta_path = None
    selected_prefit_semantic_guidance = None
    selected_fit_semantic_guidance = None
    prefit_semantic_overlay_path = None
    selected_fit_evidence_preview_path = None

    source_debug_t0 = time.perf_counter()
    if SAVE_SV_RGB_PER_WALL or SAVE_ARTIFACT_CONTACT_SHEET:
        sv_jpg_name = (
            f"sv__{geojson_base}__{facade_tag}"
            f"__selected_native_source__pano_{rec['pano_id']}"
            f"__hdg_{int(round(heading))}__pit_{int(round(pitch))}"
            f"__fov_{int(round(fov_deg))}.jpg"
        )
        sv_jpg_path = os.path.join(per_building_out, sv_jpg_name)
        img_rgb.save(sv_jpg_path, quality=95)

        for src_i, src in enumerate(source_result.get("sources", [])):
            candidate_prefit_guidance = src.get(
                "depth_global_fit_semantic_guidance",
                src.get("depth_global_prefit_semantic_guidance"),
            )
            if candidate_prefit_guidance is not None:
                candidate_semantic_path = Path(
                    per_building_out,
                    (
                        f"{geojson_base}__{facade_tag}"
                        f"__source_pano{src_i:02d}"
                        "__prefit_semantic_guidance.png"
                    ),
                )
                try:
                    candidate_semantic_overlay = (
                        create_prefit_semantic_guidance_overlay(
                            np.asarray(src["img"].convert("RGB")),
                            candidate_prefit_guidance,
                        )
                    )
                    Image.fromarray(
                        candidate_semantic_overlay,
                        mode="RGB",
                    ).save(candidate_semantic_path)
                    src["depth_global_prefit_semantic_guidance_png"] = (
                        candidate_semantic_path.name
                    )
                except Exception as candidate_semantic_exc:
                    src["depth_global_prefit_semantic_overlay_error"] = (
                        f"{type(candidate_semantic_exc).__name__}: "
                        f"{candidate_semantic_exc}"
                    )
            src_overlay_uv, _src_overlay_depth = project_points_world_to_image(
                outline_xyz,
                src["K"],
                src["Rwc"],
                src["C"],
                clip_behind=False,
            )
            if src_overlay_uv.shape[0] >= 3 and np.isfinite(src_overlay_uv).all():
                src_overlay_name = f"{geojson_base}__{facade_tag}__source_pano{src_i:02d}_overlay.png"
                src_overlay_path = os.path.join(per_building_out, src_overlay_name)
                _save_candidate_projection_screening_overlay(
                    src,
                    src_overlay_uv,
                    src_overlay_path,
                )
                src["depth_global_projection_overlay_png"] = Path(
                    src_overlay_path
                ).name
                if bool(src.get("selected_for_processing", False)):
                    _mark_selected_candidate_overlay(src_overlay_path)
        if stage_timer is not None:
            stage_timer.record(
                f"{facade_tag} / save source debug artifacts",
                time.perf_counter() - source_debug_t0,
            )

    source_candidate_meta = [
        {
            "source_index": int(source_index),
            "pano_id": str(s["rec"].get("pano_id", "")),
            "pano_lat": float(s["rec"].get("lat", 0.0)),
            "pano_lng": float(s["rec"].get("lng", 0.0)),
            "pano_copyright": str(s["rec"].get("copyright", "")),
            "pano_date": s["rec"].get("date"),
            "imagery_provider": str(s["rec"].get("imagery_provider", "unknown")),
            "search_source": str(s["rec"].get("search_source", "")),
            "camera_utm_xyz": [float(v) for v in np.asarray(s["camera_xyz"], dtype=np.float64).tolist()],
            "camera_elevation": dict(s.get("camera_elevation") or {}),
            "heading_deg": float(s["heading"]),
            "heading_reference": "true_north_google_request",
            "projection_heading_deg": float(s.get("projection_heading", s["heading"])),
            "projection_heading_reference": "source_crs_grid_north",
            "meridian_convergence_deg": float(s.get("meridian_convergence_deg", 0.0)),
            "pitch_deg": float(s["pitch"]),
            "fov_deg": float(s["fov"]),
            "u_clamped": float(s.get("u_clamped", 0.0)),
            "candidate_selection_origin": str(s.get("candidate_selection_origin", "unspecified")),
            "candidate_selection_origins": list(s.get("candidate_selection_origins", [])),
            "candidate_forward_m": float(s.get("candidate_forward_m", 0.0)),
            "candidate_frontality": float(s.get("candidate_frontality", 0.0)),
            "candidate_is_fallback": bool(s.get("candidate_is_fallback", False)),
            "legacy_wall_prism": bool(s.get("legacy_wall_prism", False)),
            "legacy_wall_framing": bool(s.get("legacy_wall_framing", False)),
            "selected_for_processing": bool(s.get("selected_for_processing", False)),
            "source_selection_rank": int(s.get("source_selection_rank", source_index + 1)),
            "target_model_visibility_available": bool(
                s.get("target_model_visibility_available", False)
            ),
            "target_model_visibility_reason": str(
                s.get("target_model_visibility_reason", "")
            ),
            "target_self_visibility_fraction": (
                None
                if s.get("target_self_visibility_fraction") is None
                else float(s["target_self_visibility_fraction"])
            ),
            "target_usable_visibility_fraction": float(
                s.get("target_usable_visibility_fraction", 0.0)
            ),
            "target_net_visibility_fraction": float(
                s.get("target_net_visibility_fraction", 0.0)
            ),
            "target_fully_visible": bool(s.get("target_fully_visible", False)),
            "target_depth_pixel_count": int(s.get("target_depth_pixel_count", 0)),
            "target_visible_pixel_count": int(s.get("target_visible_pixel_count", 0)),
            "target_occluded_pixel_count": int(s.get("target_occluded_pixel_count", 0)),
            "target_visibility_render_size_px": (
                list(s["target_visibility_render_size_px"])
                if s.get("target_visibility_render_size_px") is not None else None
            ),
            "target_model_visibility_overlay_png": None,
            "depth_global_fit_evaluated_before_selection": bool(
                s.get("depth_global_fit_evaluated_before_selection", False)
            ),
            "depth_global_fit_applied": bool(s.get("depth_global_fit_applied", False)),
            "depth_global_fit_reason": str(
                s.get("depth_global_fit_reason", "not_evaluated")
            ),
            "depth_global_score_improvement": float(
                s.get("depth_global_score_improvement", 0.0)
            ),
            "depth_global_candidate_usable": bool(
                s.get("depth_global_candidate_usable", True)
            ),
            "depth_global_candidate_rejection_reason": s.get(
                "depth_global_candidate_rejection_reason"
            ),
            "depth_global_target_visibility": dict(
                s.get("depth_global_target_visibility") or {}
            ),
            "depth_global_sam3_skipped": bool(
                s.get("depth_global_sam3_skipped", False)
            ),
            "depth_global_sam3_skip_reason": s.get(
                "depth_global_sam3_skip_reason"
            ),
            "depth_global_prefit_semantic_guidance": (
                dict(s.get("depth_global_prefit_semantic_metadata") or {})
                or None
            ),
            "depth_global_prefit_semantic_guidance_png": s.get(
                "depth_global_prefit_semantic_guidance_png"
            ),
            "depth_global_projection_overlay_geometry": s.get(
                "depth_global_projection_overlay_geometry",
                "visible_real_whole_model_edges_or_filtered_depth_fallback",
            ),
            "depth_global_projection_overlay_png": s.get(
                "depth_global_projection_overlay_png"
            ),
            "depth_global_selection_H": np.asarray(
                s.get("selection_projection_H", np.eye(3)),
                dtype=np.float64,
            ).astype(float).tolist(),
            "external_building_occlusion_available": bool(
                s.get("external_building_occlusion_available", False)
            ),
            "external_building_occlusion_fraction": (
                float(s.get("external_building_occlusion_fraction", 0.0))
                if s.get("external_building_occlusion_available", False)
                else None
            ),
            "external_building_clear": bool(s.get("external_building_clear", False)),
            "external_building_candidate_blockers": list(
                s.get("external_building_candidate_blockers", [])
            ),
            "external_building_candidate_blocker_terrain": dict(
                s.get("external_building_candidate_blocker_terrain", {})
            ),
            "external_building_blocker_terrain_source": str(
                s.get(
                    "external_building_blocker_terrain_source",
                    "not_available",
                )
            ),
            "external_building_occlusion_reason": str(
                s.get("external_building_occlusion_reason", "not_evaluated")
            ),
            "full_frame_coverage": bool(s.get("full_frame_coverage", False)),
            "projected_coverage_fraction": float(s.get("projected_coverage_fraction", 0.0)),
            "projected_visible_area_px2": float(s.get("projected_area_px2", 0.0)),
            "projected_area_fraction": float(s.get("projected_area_fraction", 0.0)),
            "min_projected_span_px": float(s.get("min_projected_span_px", 0.0)),
            "nondegenerate_projection": bool(s.get("nondegenerate_projection", False)),
            "uses_near_plane_clipped_projection": bool(
                s.get("uses_near_plane_clipped_projection", False)
            ),
            "projection_score": list(s["projection_score"]) if s.get("projection_score") is not None else None,
            "image_space_wireframe_fit": s.get("wireframe_fit"),
            "effective_camera_parameter_fit": s.get("effective_camera_fit"),
            "wireframe_fit_overlay_png": None,
            "street_view_url": _mask_key(str(s.get("url", ""))),
        }
        for source_index, s in enumerate(source_result.get("sources", []))
    ]

    print(
        f"[{facade_tag}] fetched and projected {len(urls_fetched)} Street View image(s) "
        f"for {len(group_records)} wall fragments | mode={source_mode}."
    )
    for u in urls_fetched:
        safe_u = u.split("&key=")[0] + "&key=****" if "&key=" in u else u
        print("         URL:", safe_u)

    wireframe_fit_overlay_path = Path(
        per_building_out,
        f"{geojson_base}__{facade_tag}__selected_source_wireframe_fit_overlay.png",
    )
    selected_source_fit_overlay = source_result.get("wireframe_fit_overlay")
    if (
        requested_alignment_mode == "wall_only"
        and selected_source_fit_overlay is not None
        and (SAVE_FACADE_GROUP_DEBUG_PNG or SAVE_ARTIFACT_CONTACT_SHEET)
    ):
        selected_source_fit_overlay.save(wireframe_fit_overlay_path)

    _save_temp_global_wall_group_image_projection(
        per_building_out=per_building_out,
        geojson_base=geojson_base,
        facade_tag=facade_tag,
        group_id=group_id,
        group_records=group_records,
        outline_xyz=outline_xyz,
        raw_sources=source_result.get("sources", []),
    )

    if (
        bool(globals().get("SAVE_MODEL_DEPTH_MAPS", True))
        or bool(globals().get("ENABLE_DEPTH_AWARE_REGION_FIT", False))
        or bool(
            requested_alignment_mode == "depth_global"
            and globals().get("ENABLE_MODEL_DEPTH_BOUNDARY_FIT", False)
        )
    ):
        try:
            sources = list(source_result.get("sources", []))
            selected_source_index = int(source_result.get("selected_source_index", 0))
            selected_source = (
                sources[selected_source_index]
                if 0 <= selected_source_index < len(sources)
                else None
            )
            if selected_source is not None:
                depth_K = np.asarray(selected_source["K"], dtype=np.float64)
                depth_R = np.asarray(selected_source["Rwc"], dtype=np.float64)
                depth_H = np.asarray(
                    source_result.get(
                        "selected_source_raw_to_processing_image_H",
                        np.eye(3),
                    ),
                    dtype=np.float64,
                )
                depth_C = np.asarray(selected_source["C"], dtype=np.float64)
                depth_camera_metadata = {
                    "mode": "raw_selected_camera_before_any_wall_local_wireframe_fit",
                    "selected_source_index": int(selected_source_index),
                    "camera_utm_xyz": [float(v) for v in depth_C.tolist()],
                    "heading_deg": float(selected_source.get("heading", 0.0)),
                    "heading_reference": "true_north_google_request",
                    "projection_heading_deg": float(selected_source.get(
                        "projection_heading",
                        selected_source.get("heading", 0.0),
                    )),
                    "projection_heading_reference": "source_crs_grid_north",
                    "meridian_convergence_deg": float(selected_source.get(
                        "meridian_convergence_deg",
                        0.0,
                    )),
                    "pitch_deg": float(selected_source.get("pitch", 0.0)),
                    "fov_deg": float(selected_source.get("fov", 0.0)),
                    "independent_of_wall_local_fit": True,
                }
                full_depth = None
                if preselected_full_model_depth is not None:
                    candidate_depth = np.asarray(
                        preselected_full_model_depth,
                        dtype=np.float32,
                    )
                    if candidate_depth.shape == (img_rgb.height, img_rgb.width):
                        full_depth = candidate_depth.copy()
                if full_depth is None:
                    full_depth = _render_model_depth_view(
                        meshes_named=meshes_named or [],
                        K=depth_K,
                        R_wc=depth_R,
                        C=depth_C,
                        source_image_size=selected_source["img"].size,
                        output_image_size=img_rgb.size,
                        image_to_output_H=depth_H,
                    )
                if full_depth is None:
                    raise RuntimeError("Whole-model depth rendering returned no image.")
                # Adjacent-wall visibility is compared in the raw camera
                # raster.  A depth map rendered through ``depth_H`` is already
                # in processing-image coordinates and must not be compared to
                # an adjacent-only raw render.  Keep the known raw candidate
                # depth above; otherwise let the side helper render raw depth.
                if (
                    selected_full_model_depth_for_sides is None
                    and np.allclose(depth_H, np.eye(3), atol=1.0e-9)
                    and tuple(full_depth.shape)
                    == (
                        int(selected_source["img"].height),
                        int(selected_source["img"].width),
                    )
                ):
                    selected_full_model_depth_for_sides = np.asarray(
                        full_depth, dtype=np.float32
                    ).copy()

                depth_artifacts = _save_model_depth_map_artifacts(
                    per_building_out=per_building_out,
                    prefix_name=f"{geojson_base}__{facade_tag}__model_depth",
                    meshes_named=meshes_named or [],
                    K=depth_K,
                    R_wc=depth_R,
                    C=depth_C,
                    source_image_size=selected_source["img"].size,
                    output_image_size=img_rgb.size,
                    image_to_output_H=depth_H,
                    camera_metadata=depth_camera_metadata,
                    precomputed_output_depth=full_depth,
                )

                if (
                    requested_alignment_mode == "depth_global"
                    and bool(globals().get("ENABLE_MODEL_DEPTH_BOUNDARY_FIT", False))
                ):
                    image_bgr = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)
                    selected_osm_masked_refit = bool(
                        selected_external_building_mask is not None
                    )
                    cached_prefit_guidance = source_result.get(
                        "selected_candidate_prefit_semantic_guidance"
                    )
                    cached_fit_guidance = source_result.get(
                        "selected_candidate_fit_semantic_guidance"
                    )
                    if (
                        cached_prefit_guidance is None
                        and isinstance(selected_source, dict)
                    ):
                        cached_prefit_guidance = selected_source.get(
                            "depth_global_prefit_semantic_guidance"
                        )
                    if (
                        cached_fit_guidance is None
                        and isinstance(selected_source, dict)
                    ):
                        cached_fit_guidance = selected_source.get(
                            "depth_global_fit_semantic_guidance"
                        )
                    if cached_fit_guidance is None:
                        cached_fit_guidance = cached_prefit_guidance
                    selected_prefit_semantic_guidance = cached_prefit_guidance
                    selected_fit_semantic_guidance = cached_fit_guidance
                    if cached_prefit_guidance is not None:
                        cached_shape = np.asarray(
                            cached_prefit_guidance.get(
                                "raw_projection_mask",
                                np.zeros((0, 0), dtype=bool),
                            ),
                            dtype=bool,
                        ).shape
                        if cached_shape != image_bgr.shape[:2]:
                            cached_prefit_guidance = None
                            cached_fit_guidance = None
                            selected_prefit_semantic_guidance = None
                            selected_fit_semantic_guidance = None
                    if cached_fit_guidance is not None:
                        fit_cached_shape = np.asarray(
                            cached_fit_guidance.get(
                                "raw_projection_mask",
                                np.zeros((0, 0), dtype=bool),
                            ),
                            dtype=bool,
                        ).shape
                        if fit_cached_shape != image_bgr.shape[:2]:
                            cached_fit_guidance = None
                            selected_fit_semantic_guidance = None
                    if selected_fit_semantic_guidance is None:
                        selected_fit_semantic_guidance = (
                            selected_prefit_semantic_guidance
                        )
                    rerun_prefit_semantics = cached_prefit_guidance is None
                    if rerun_prefit_semantics:
                        try:
                            selected_prefit_semantic_guidance = (
                                _run_model_depth_prefit_semantic_guidance(
                                    processor=processor,
                                    image_rgb=img_rgb,
                                    raw_projection_mask=(
                                        np.isfinite(full_depth)
                                        & (full_depth > 0.0)
                                    ),
                                    target_wall_projection_mask=(
                                        selected_source.get(
                                            "depth_global_target_wall_projection_mask"
                                        )
                                    ),
                                    external_exclusion_mask=(
                                        selected_source.get(
                                            "external_building_fit_exclusion_mask"
                                        )
                                    ),
                                    stage=(
                                        "selected_processing_image_before_global_depth_fit"
                                    ),
                                )
                            )
                            selected_fit_semantic_guidance = (
                                selected_prefit_semantic_guidance
                            )
                        except Exception as semantic_image_exc:
                            print(
                                f"[{facade_tag}] selected pre-fit semantic "
                                f"guidance unavailable ({semantic_image_exc})."
                            )
                            selected_prefit_semantic_guidance = (
                                cached_prefit_guidance
                            )
                    elif selected_prefit_semantic_guidance is not None:
                        # The selected image and its pixel coordinates did not
                        # change. OSM only supplies another evidence mask, so
                        # reuse the candidate embedding/masks for the refit.
                        selected_prefit_semantic_guidance = dict(
                            selected_prefit_semantic_guidance
                        )
                        reused_metadata = dict(
                            selected_prefit_semantic_guidance.get(
                                "metadata",
                                {},
                            )
                        )
                        reused_metadata.update({
                            "reused_from_candidate_preselection": True,
                            "reused_for_selected_osm_refit": bool(
                                selected_osm_masked_refit
                            ),
                            "selected_source_second_segmentation_run": False,
                        })
                        selected_prefit_semantic_guidance["metadata"] = (
                            reused_metadata
                        )
                        if selected_fit_semantic_guidance is None:
                            selected_fit_semantic_guidance = (
                                selected_prefit_semantic_guidance
                            )
                        elif (
                            selected_fit_semantic_guidance
                            is not selected_prefit_semantic_guidance
                        ):
                            selected_fit_semantic_guidance = dict(
                                selected_fit_semantic_guidance
                            )
                    selected_fit_evidence = (
                        _combine_model_depth_fit_evidence(
                            selected_fit_semantic_guidance,
                            image_bgr.shape[:2],
                            external_exclusion_mask=(
                                selected_external_building_mask
                                if selected_osm_masked_refit else None
                            ),
                            background_aware=bool(
                                isinstance(
                                    selected_fit_semantic_guidance,
                                    Mapping,
                                )
                                and selected_fit_semantic_guidance.get(
                                    "background_aware_active",
                                    False,
                                )
                            ),
                        )
                    )
                    valid_image_evidence_mask = selected_fit_evidence[
                        "valid_evidence_mask"
                    ]
                    reuse_preselected_depth_fit = bool(
                        isinstance(preselected_depth_boundary_fit_result, dict)
                        and preselected_depth_boundary_fit_result.get("homography") is not None
                        and not selected_osm_masked_refit
                    )
                    semantic_boundary_geometry = None
                    if not reuse_preselected_depth_fit and bool(globals().get(
                        "MODEL_DEPTH_BOUNDARY_USE_SEMANTIC_GUIDES",
                        True,
                    )):
                        try:
                            semantic_boundary_geometry = project_semantic_model_boundary_edges(
                                model_edges_xyz_by_class=model_boundary_edges_xyz or {},
                                K=depth_K,
                                R_wc=depth_R,
                                C=depth_C,
                                full_model_depth=full_depth,
                                image_to_output_H=depth_H,
                                near_m=float(globals().get("MODEL_DEPTH_NEAR_M", 0.05)),
                                sample_step_px=float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_SAMPLE_STEP_PX", 2.0,
                                )),
                                silhouette_tolerance_px=float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_SILHOUETTE_TOLERANCE_PX", 4.0,
                                )),
                                depth_search_radius_px=int(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_SEARCH_RADIUS_PX", 2,
                                )),
                                depth_tolerance_m=float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_TOLERANCE_M", 0.35,
                                )),
                                depth_relative_tolerance=float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_DEPTH_RELATIVE_TOLERANCE", 0.03,
                                )),
                                maximum_visibility_gap_samples=int(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_MAX_GAP_SAMPLES", 2,
                                )),
                                minimum_visible_run_px=float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_SEMANTIC_MIN_RUN_PX", 8.0,
                                )),
                            )
                        except Exception as semantic_exc:
                            print(
                                f"[{facade_tag}] semantic model-edge guides unavailable; "
                                f"using depth silhouette fallback: {semantic_exc}"
                            )
                    if reuse_preselected_depth_fit:
                        depth_boundary_fit_result = preselected_depth_boundary_fit_result
                    else:
                        depth_boundary_fit_result = fit_depth_silhouette_to_image(
                            image_bgr=image_bgr,
                            full_model_depth=full_depth,
                            raw_wall_outline_px=raw_uv_outline,
                            wall_local_fit_outline_px=wall_only_uv_outline,
                            fit_config=_model_depth_boundary_fit_config(
                                semantic_target_supported=bool(
                                    selected_source.get(
                                        "depth_global_semantic_fit_anchor_supported",
                                        False,
                                    )
                                ),
                            ),
                            minimum_area_px=int(globals().get(
                                "MODEL_DEPTH_BOUNDARY_FIT_MIN_AREA_PX", 350,
                            )),
                            minimum_component_fraction=float(globals().get(
                                "MODEL_DEPTH_BOUNDARY_FIT_MIN_COMPONENT_FRACTION", 0.02,
                            )),
                            contour_epsilon_px=float(globals().get(
                                "MODEL_DEPTH_BOUNDARY_FIT_CONTOUR_EPSILON_PX", 1.5,
                            )),
                            maximum_points=int(globals().get(
                                "MODEL_DEPTH_BOUNDARY_FIT_MAX_POINTS", 240,
                            )),
                            semantic_boundary_geometry=semantic_boundary_geometry,
                            semantic_class_weights={
                                "roof": float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_ROOF_WEIGHT", 3.0,
                                )),
                                "wall": float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_WALL_WEIGHT", 2.0,
                                )),
                                "base": float(globals().get(
                                    "MODEL_DEPTH_BOUNDARY_BASE_WEIGHT", 0.35,
                                )),
                            },
                            valid_image_evidence_mask=valid_image_evidence_mask,
                            semantic_valid_image_evidence_mask=(
                                selected_fit_evidence.get(
                                    "semantic_valid_evidence_mask"
                                )
                            ),
                            semantic_image_boundary_maps=selected_fit_evidence[
                                "boundary_maps"
                            ],
                            semantic_image_guidance_metadata=selected_fit_evidence[
                                "metadata"
                            ],
                        )
                        if selected_osm_masked_refit:
                            depth_boundary_fit_result = (
                                _finalize_selected_osm_masked_depth_refit(
                                    depth_boundary_fit_result,
                                    raw_wall_outline_px=raw_uv_outline,
                                    exclusion_mask=selected_external_building_mask,
                                    valid_evidence_mask=valid_image_evidence_mask,
                                )
                            )
                    depth_boundary_fit_info = depth_boundary_fit_metadata(
                        depth_boundary_fit_result
                    )
                    depth_boundary_fit_info.update({
                        "enabled": True,
                        "raw_camera_to_processing_image_H": depth_H.astype(float).tolist(),
                        "requested_downstream_alignment": requested_alignment_mode,
                        "reused_candidate_preselection_fit": bool(
                            reuse_preselected_depth_fit
                        ),
                        "selected_source_osm_refit": selected_osm_masked_refit,
                        "selected_source_osm_refit_used_lr_style_side_exclusion": (
                            selected_osm_masked_refit
                        ),
                        "selected_source_osm_refit_fallback_to_preselection": bool(
                            depth_boundary_fit_result.get(
                                "selected_source_osm_refit_fallback_to_preselection",
                                False,
                            )
                        ),
                    })

                    boundary_overlay_path = Path(
                        per_building_out,
                        f"{geojson_base}__{facade_tag}__whole_model_depth_boundary_fit.png",
                    )
                    raw_silhouette_path = Path(
                        per_building_out,
                        f"{geojson_base}__{facade_tag}__model_depth_silhouette_mask.png",
                    )
                    boundary_meta_path = Path(
                        per_building_out,
                        f"{geojson_base}__{facade_tag}__model_depth_boundary_fit_meta.json",
                    )
                    if (
                        selected_fit_evidence.get("overlay_guidance") is not None
                        and (
                            bool(globals().get(
                                "SAVE_MODEL_DEPTH_PREFIT_SEMANTIC_DEBUG",
                                True,
                            ))
                            or SAVE_ARTIFACT_CONTACT_SHEET
                        )
                    ):
                        prefit_semantic_overlay_path = Path(
                            per_building_out,
                            (
                                f"{geojson_base}__{facade_tag}"
                                "__model_depth_prefit_semantic_guidance.png"
                            ),
                        )
                        semantic_overlay_rgb = (
                            create_prefit_semantic_guidance_overlay(
                                np.asarray(img_rgb.convert("RGB")),
                                selected_fit_evidence["overlay_guidance"],
                            )
                        )
                        Image.fromarray(
                            semantic_overlay_rgb,
                            mode="RGB",
                        ).save(prefit_semantic_overlay_path)

                    if valid_image_evidence_mask is not None:
                        selected_fit_evidence_preview_path = Path(
                            per_building_out,
                            (
                                f"{geojson_base}__{facade_tag}"
                                "__selected_depth_global_fit_evidence_preview.png"
                            ),
                        )
                        valid_fit_evidence = np.asarray(
                            valid_image_evidence_mask,
                            dtype=bool,
                        )
                        if valid_fit_evidence.shape != image_bgr.shape[:2]:
                            raise ValueError(
                                "Selected depth-global evidence preview mask "
                                "must match the processing image."
                            )
                        Image.fromarray(
                            _external_building_removal_preview_rgb(
                                img_rgb,
                                ~valid_fit_evidence,
                            ),
                            mode="RGB",
                        ).save(selected_fit_evidence_preview_path)

                    if bool(globals().get("SAVE_MODEL_DEPTH_BOUNDARY_FIT_DEBUG", True)) or SAVE_ARTIFACT_CONTACT_SHEET:
                        boundary_overlay_image_bgr = image_bgr
                        if selected_osm_masked_refit:
                            boundary_overlay_image_bgr = cv2.cvtColor(
                                _external_building_removal_preview_rgb(
                                    img_rgb,
                                    selected_external_building_mask,
                                ),
                                cv2.COLOR_RGB2BGR,
                            )
                        boundary_overlay = create_depth_boundary_fit_overlay(
                            boundary_overlay_image_bgr,
                            depth_boundary_fit_result,
                            _model_depth_boundary_fit_config(),
                        )
                        cv2.imwrite(str(boundary_overlay_path), boundary_overlay)
                        silhouette_shift_overlay = create_depth_silhouette_shift_overlay(
                            depth_boundary_fit_result,
                            _model_depth_boundary_fit_config(),
                        )
                        cv2.imwrite(str(raw_silhouette_path), silhouette_shift_overlay)

                    with open(boundary_meta_path, "w", encoding="utf-8") as f:
                        json.dump(depth_boundary_fit_info, f, ensure_ascii=False, indent=2)

                    depth_boundary_artifacts = {
                        "raw_silhouette_mask_png": raw_silhouette_path.name,
                        "boundary_fit_overlay_png": boundary_overlay_path.name,
                        "whole_model_depth_boundary_fit_png": boundary_overlay_path.name,
                        "boundary_fit_meta_json": boundary_meta_path.name,
                        "prefit_semantic_guidance_png": (
                            prefit_semantic_overlay_path.name
                            if prefit_semantic_overlay_path is not None
                            else None
                        ),
                        "selected_fit_evidence_preview_png": (
                            selected_fit_evidence_preview_path.name
                            if selected_fit_evidence_preview_path is not None
                            else None
                        ),
                    }
                    transform = depth_boundary_fit_result.get("transform", {})
                    status = "accepted" if depth_boundary_fit_result.get("applied") else "candidate only"
                    expected_downstream = (
                        "depth-global"
                        if requested_alignment_mode == "depth_global"
                        and depth_boundary_fit_result.get("applied")
                        else "wall-only"
                    )
                    print(
                        f"[{facade_tag}] whole-model depth boundary fit {status} | "
                        f"scale={float(transform.get('scale', 1.0)):.4f}, "
                        f"tx={float(transform.get('tx', 0.0)):.1f}px, "
                        f"ty={float(transform.get('ty', 0.0)):.1f}px, "
                        f"gain={float(depth_boundary_fit_result.get('score_improvement', 0.0)):.4f} | "
                        f"downstream={expected_downstream}"
                    )
        except Exception as exc:
            print(f"[{facade_tag}] model depth map failed: {exc}")
            if (
                requested_alignment_mode == "depth_global"
                and bool(globals().get("ENABLE_MODEL_DEPTH_BOUNDARY_FIT", False))
            ):
                depth_boundary_fit_info = {
                    "enabled": True,
                    "applied": False,
                    "reason": f"fit_failed: {exc}",
                    "uses_segmentation": False,
                    "downstream_authoritative": False,
                    "requested_downstream_alignment": requested_alignment_mode,
                }

    facade_alignment_selection = select_facade_alignment(
        requested_mode=requested_alignment_mode,
        wall_only_outline_px=wall_only_uv_outline,
        wall_only_rect_px=wall_only_uv_rect,
        raw_outline_px=raw_uv_outline,
        raw_rect_px=raw_uv_rect,
        depth_fit_result=depth_boundary_fit_result,
    )
    uv_outline = np.asarray(facade_alignment_selection["outline_px"], dtype=np.float64)
    uv_rect = np.asarray(facade_alignment_selection["rect_px"], dtype=np.float64)
    facade_alignment_info = facade_alignment_metadata(facade_alignment_selection)
    effective_alignment_mode = str(facade_alignment_selection["effective_mode"])
    depth_boundary_fit_info.update({
        "downstream_authoritative": effective_alignment_mode == "depth_global",
        "production_wall_projection_unchanged": effective_alignment_mode == "wall_only",
        "facade_alignment": facade_alignment_info,
    })
    if boundary_meta_path is not None:
        with open(boundary_meta_path, "w", encoding="utf-8") as f:
            json.dump(depth_boundary_fit_info, f, ensure_ascii=False, indent=2)

    if facade_alignment_selection.get("fallback"):
        print(
            f"[{facade_tag}] requested depth-global alignment unavailable; "
            f"using wall-only ({facade_alignment_selection.get('fallback_reason')})."
        )
    else:
        print(f"[{facade_tag}] downstream facade alignment: {effective_alignment_mode}.")

    selected_alignment_overlay_path = Path(
        per_building_out,
        f"{geojson_base}__{facade_tag}__selected_{effective_alignment_mode}_processing_overlay.png",
    )
    if (
        effective_alignment_mode != "depth_global"
        and (SAVE_RAW_OVERLAY_PNG or SAVE_ARTIFACT_CONTACT_SHEET)
    ):
        save_overlay_matplotlib(
            img_rgb,
            uv_outline,
            selected_alignment_overlay_path,
            title=(
                f"Facade {facade_tag} - selected {effective_alignment_mode} alignment - "
                f"heading {heading:.1f}, pitch {pitch:.1f}, fov {fov_deg:.1f}"
            ),
        )

    segmentation_search_buffer_px = int(LR_BAND_BUFFER_PX)
    if bool(globals().get("ENABLE_DEPTH_AWARE_REGION_FIT", False)):
        segmentation_search_buffer_px = max(
            segmentation_search_buffer_px,
            int(round(float(globals().get("DEPTH_AWARE_REGION_FIT_SEARCH_MARGIN_PX", 120.0)))),
        )
    lr_rgba, band_poly, band_bbox = build_lr_band_rgba(
        img_rgb,
        uv_rect,
        segmentation_search_buffer_px,
    )
    if lr_rgba is None:
        print(f"[{facade_tag}] LR-band failed - grouped wall not textured.")
        return {}

    if SAVE_LR_OVERLAY_PNG or SAVE_ARTIFACT_CONTACT_SHEET:
        lr_overlay_path = os.path.join(
            per_building_out,
            f"{geojson_base}__{facade_tag}__lr_band_overlay.png"
        )
        save_with_overlay(lr_rgba, uv_outline, lr_overlay_path)

    W, H = lr_rgba.size
    r, g, b, _lr_alpha = lr_rgba.split()
    reuse_prefit_semantic_mask_enabled = bool(globals().get(
        "ENABLE_PREFIT_SEMANTIC_TEXTURE_MASK_REUSE",
        True,
    ))
    semantic_target_projection_mask = _polygon_to_mask(
        H, W, uv_outline
    )
    if (
        selected_prefit_semantic_guidance is None
        and (
            reuse_prefit_semantic_mask_enabled
            or bool(globals().get(
                "ENABLE_FACADE_SIDE_SEMANTIC_RECOVERY", True,
            ))
        )
    ):
        try:
            selected_prefit_semantic_guidance = (
                _run_model_depth_prefit_semantic_guidance(
                    processor=processor,
                    image_rgb=img_rgb,
                    raw_projection_mask=semantic_target_projection_mask,
                    target_wall_projection_mask=(
                        semantic_target_projection_mask
                    ),
                    external_exclusion_mask=(
                        selected_external_building_mask
                    ),
                    stage=(
                        "selected_source_for_facade_side_and_texture_semantics"
                    ),
                )
            )
            selected_fit_semantic_guidance = (
                selected_prefit_semantic_guidance
            )
        except Exception as semantic_side_exc:
            print(
                f"[{facade_tag}] selected facade-side semantic guidance "
                f"unavailable ({semantic_side_exc})."
            )
    semantic_reuse_t0 = time.perf_counter()
    semantic_reuse_overlay_path = None
    refinement_info = {
        "enabled": reuse_prefit_semantic_mask_enabled,
        "stage": "selected_source_full_image_before_global_depth_fit",
        "accepted_for_reuse": False,
        "reason": "pending_prefit_semantic_mask_reuse",
        "second_segmentation_inference_run": False,
    }
    pred_full_clean = np.zeros((H, W), dtype=bool)
    if selected_prefit_semantic_guidance is not None:
        selected_target = selected_prefit_semantic_guidance.get(
            "target_semantic_mask"
        )
        if selected_target is not None:
            selected_target = np.asarray(selected_target, dtype=bool)
            if selected_target.shape == (H, W):
                pred_full_clean = selected_target.copy()
    lr_alpha_gate_info = {
        "enabled": False,
        "mode": "full_canvas_prefit_semantic_mask_reuse",
        "reason": "no_secondary_crop_or_segmentation",
        "selected_px_before_gate": int(pred_full_clean.sum()),
        "kept_by_raw_lr_alpha_px": int(pred_full_clean.sum()),
        "rescued_selected_px": 0,
        "removed_selected_px": 0,
        "wall_margin_px": 0,
    }
    depth_aware_region_fit_result = None
    legacy_region_fit_requested = bool(globals().get("ENABLE_DEPTH_AWARE_REGION_FIT", False))
    depth_aware_region_fit_info = {
        "enabled": legacy_region_fit_requested and bool(pred_full_clean.any()),
        "applied": False,
        "reason": (
            "no_reused_prefit_target_mask"
            if legacy_region_fit_requested and not pred_full_clean.any()
            else "disabled"
        ),
        "segmentation_search_buffer_px": int(segmentation_search_buffer_px),
        "segmentation_source": "reused_full_image_prefit_semantics",
    }
    depth_aware_region_fit_overlay_path = Path(
        per_building_out,
        f"{geojson_base}__{facade_tag}__depth_aware_region_fit_overlay.png",
    )
    region_fit_H = np.eye(3, dtype=np.float64)

    if depth_aware_region_fit_info["enabled"]:
        projected_group_mask = _polygon_to_mask(H, W, uv_outline)
        projected_group_mask_source = "projected_group_outline_fallback"
        if depth_fit_context is not None:
            full_depth = depth_fit_context.get("full_depth")
            group_depth = depth_fit_context.get("group_depth")
            if (
                full_depth is not None
                and group_depth is not None
                and np.asarray(full_depth).shape == (H, W)
                and np.asarray(group_depth).shape == (H, W)
            ):
                try:
                    visible_depth_mask = visible_group_mask_from_depth(
                        full_depth,
                        group_depth,
                        absolute_tolerance_m=float(globals().get(
                            "DEPTH_AWARE_REGION_FIT_DEPTH_TOLERANCE_M",
                            0.08,
                        )),
                    )
                    if int(visible_depth_mask.sum()) >= 350:
                        projected_group_mask = visible_depth_mask
                        projected_group_mask_source = "full_model_zbuffer_visible_group"
                except Exception as exc:
                    print(f"[{facade_tag}] visible group depth mask failed: {exc}")

        try:
            with _timer_stage(stage_timer, f"{facade_tag} / depth-aware segmentation region fit"):
                depth_aware_region_fit_result = fit_depth_aware_segmentation_region(
                    segmentation_mask=pred_full_clean,
                    projected_group_mask=projected_group_mask,
                    outline_points_px=uv_outline,
                    image_bgr=cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR),
                    full_model_depth=(
                        depth_fit_context.get("full_depth")
                        if depth_fit_context is not None else None
                    ),
                    config=_depth_aware_region_fit_config(),
                )
            depth_aware_region_fit_info = depth_aware_region_fit_metadata(
                depth_aware_region_fit_result
            )
            depth_aware_region_fit_info.update({
                "enabled": True,
                "projected_group_mask_source": projected_group_mask_source,
                "segmentation_search_buffer_px": int(segmentation_search_buffer_px),
            })
            if depth_aware_region_fit_result.get("applied"):
                region_fit_H = np.asarray(
                    depth_aware_region_fit_result["homography"],
                    dtype=np.float64,
                )
                uv_outline = apply_H(uv_outline, region_fit_H)
                uv_rect = apply_H(uv_rect, region_fit_H)

            transform = depth_aware_region_fit_result.get("transform", {})
            status = "applied" if depth_aware_region_fit_result.get("applied") else "kept wireframe fit"
            print(
                f"[{facade_tag}] depth-aware region fit {status} | "
                f"scale={float(transform.get('scale', 1.0)):.4f}, "
                f"tx={float(transform.get('tx', 0.0)):.1f}px, "
                f"ty={float(transform.get('ty', 0.0)):.1f}px, "
                f"gain={float(depth_aware_region_fit_result.get('score_improvement', 0.0)):.4f} | "
                f"reason={depth_aware_region_fit_result.get('reason')}"
            )
        except Exception as exc:
            depth_aware_region_fit_info = {
                "enabled": True,
                "applied": False,
                "reason": f"fit_failed: {exc}",
                "projected_group_mask_source": projected_group_mask_source,
                "segmentation_search_buffer_px": int(segmentation_search_buffer_px),
            }
            print(f"[{facade_tag}] depth-aware segmentation region fit failed: {exc}")

    if (
        depth_aware_region_fit_result is not None
        and bool(globals().get("SAVE_DEPTH_AWARE_REGION_FIT_DEBUG", True))
        and (SAVE_FACADE_GROUP_DEBUG_PNG or SAVE_ARTIFACT_CONTACT_SHEET)
    ):
        try:
            region_overlay_bgr = create_depth_aware_region_fit_overlay(
                cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR),
                depth_aware_region_fit_result,
            )
            Image.fromarray(cv2.cvtColor(region_overlay_bgr, cv2.COLOR_BGR2RGB)).save(
                depth_aware_region_fit_overlay_path
            )
        except Exception as exc:
            print(f"[{facade_tag}] depth-aware region fit overlay failed: {exc}")

    if (
        depth_aware_region_fit_result is not None
        and depth_aware_region_fit_result.get("applied")
        and depth_fit_context is not None
        and depth_fit_context.get("full_depth") is not None
    ):
        try:
            refined_full_depth = warp_depth_map_to_canvas(
                depth_fit_context["full_depth"],
                region_fit_H,
                img_rgb.size,
            )
            final_depth_H = region_fit_H @ np.asarray(
                depth_fit_context["base_image_to_output_H"],
                dtype=np.float64,
            )
            final_depth_camera_metadata = dict(depth_fit_context["camera_metadata"])
            final_depth_camera_metadata.update({
                "prefit_semantic_region_fit": depth_aware_region_fit_info,
                "depth_value_note": (
                    "Camera-forward metric depth from the anchor camera; the accepted "
                    "region correction moves the entire connected model in image space."
                ),
            })
            depth_artifacts = _save_model_depth_map_artifacts(
                per_building_out=per_building_out,
                prefix_name=f"{geojson_base}__{facade_tag}__model_depth",
                meshes_named=meshes_named or [],
                K=depth_fit_context["K"],
                R_wc=depth_fit_context["R_wc"],
                C=depth_fit_context["C"],
                source_image_size=depth_fit_context["source_image_size"],
                output_image_size=img_rgb.size,
                image_to_output_H=final_depth_H,
                camera_metadata=final_depth_camera_metadata,
                precomputed_output_depth=refined_full_depth,
            )
            depth_fit_context["full_depth"] = refined_full_depth
            depth_fit_context["base_image_to_output_H"] = final_depth_H
        except Exception as exc:
            print(f"[{facade_tag}] final region-refined model depth save failed: {exc}")

    source_side_evidence = {
        "enabled": False,
        "reason": "disabled",
        "sides": {},
        "content_extension_mask": np.zeros((H, W), dtype=bool),
    }
    if bool(globals().get("ENABLE_FACADE_SIDE_SEMANTIC_RECOVERY", True)):
        try:
            if selected_source is None:
                raise ValueError("selected_source_geometry_unavailable")
            if effective_alignment_mode == "depth_global":
                base_model_to_selected_h = np.asarray(
                    facade_alignment_selection.get("homography", np.eye(3)),
                    dtype=np.float64,
                )
            else:
                base_model_to_selected_h = np.asarray(
                    source_result.get(
                        "selected_source_raw_to_aligned_image_H",
                        np.eye(3),
                    ),
                    dtype=np.float64,
                )
            model_to_selected_h = (
                np.asarray(region_fit_H, dtype=np.float64)
                @ base_model_to_selected_h
            )
            adjacent_contexts = build_adjacent_wall_contexts(
                group_records=group_records,
                loop_records=list(loop_records or []),
                mesh_by_name=mesh_by_name,
                meshes_named=list(meshes_named or []),
                K=np.asarray(selected_source["K"], dtype=np.float64),
                R_wc=np.asarray(selected_source["Rwc"], dtype=np.float64),
                camera_xyz=np.asarray(selected_source["C"], dtype=np.float64),
                raw_image_size_wh=selected_source["img"].size,
                selected_image_size_wh=img_rgb.size,
                model_to_selected_h=model_to_selected_h,
                raw_full_depth=selected_full_model_depth_for_sides,
                side_band_px=int(globals().get(
                    "FACADE_SIDE_SOURCE_BAND_PX", 48,
                )),
                minimum_visible_fraction=float(globals().get(
                    "FACADE_SIDE_MIN_ADJACENT_VISIBLE_FRACTION", 0.08,
                )),
            )
            source_side_evidence = analyze_source_side_evidence(
                target_outline_px=uv_outline,
                semantic_guidance=selected_prefit_semantic_guidance,
                adjacent_contexts=adjacent_contexts,
                image_shape_hw=(H, W),
                external_exclusion_mask=selected_external_building_mask,
                side_band_px=int(globals().get(
                    "FACADE_SIDE_SOURCE_BAND_PX", 48,
                )),
                foreground_occlusion_ratio=float(globals().get(
                    "FACADE_SIDE_FOREGROUND_OCCLUSION_RATIO", 0.50,
                )),
            )
        except Exception as side_exc:
            source_side_evidence = {
                "enabled": False,
                "reason": f"side_evidence_failed: {side_exc}",
                "sides": {},
                "content_extension_mask": np.zeros((H, W), dtype=bool),
            }
            print(f"[{facade_tag}] source side-evidence analysis failed: {side_exc}")

    opening_sam_rows = []
    opening_sam_info = {
        "enabled": bool(globals().get(
            "ENABLE_OPENING_AWARE_RECTIFICATION", True,
        )),
        "reason": "disabled",
        "raw_instance_count": 0,
    }
    if opening_sam_info["enabled"]:
        try:
            opening_sam_rows, opening_runtime = run_opening_sam3_prompts(
                processor,
                img_rgb,
                globals().get(
                    "OPENING_AWARE_PROMPT_LIBRARY",
                    {
                        "window": (
                            "window", "building window", "shop window",
                        ),
                        "door": ("door", "building entrance door"),
                    },
                ),
                proposal_threshold=float(globals().get(
                    "OPENING_AWARE_PROPOSAL_THRESHOLD", 0.20,
                )),
            )
            opening_sam_info = {
                "enabled": True,
                "reason": "completed",
                **opening_runtime,
            }
        except Exception as opening_exc:
            opening_sam_info = {
                "enabled": True,
                "reason": f"opening_sam3_failed: {opening_exc}",
                "raw_instance_count": 0,
            }
            print(f"[{facade_tag}] SAM3 opening detection failed: {opening_exc}")

    projection_mask_full = semantic_target_projection_mask
    if not projection_mask_full.any():
        print(f"[{facade_tag}] Selected wall projection is empty on the source image.")
        return {}
    semantic_reuse_result = build_reused_prefit_facade_mask(
        selected_prefit_semantic_guidance,
        projection_mask_full,
        external_exclusion_mask=selected_external_building_mask,
        enabled=reuse_prefit_semantic_mask_enabled,
        minimum_pixels=int(globals().get(
            "PREFIT_SEMANTIC_TEXTURE_MIN_PIXELS",
            250,
        )),
        minimum_wall_coverage=float(globals().get(
            "PREFIT_SEMANTIC_TEXTURE_MIN_WALL_COVERAGE",
            0.35,
        )),
        closing_radius_px=int(globals().get(
            "PREFIT_SEMANTIC_TEXTURE_CLOSE_PX",
            2,
        )),
        maximum_hole_area_px=int(globals().get(
            "PREFIT_SEMANTIC_TEXTURE_MAX_HOLE_AREA_PX",
            900,
        )),
        maximum_hard_exclusion_fraction=float(globals().get(
            "PREFIT_SEMANTIC_TEXTURE_MAX_HARD_EXCLUSION_FRACTION",
            0.85,
        )),
    )
    retained_source_mask = np.asarray(
        semantic_reuse_result["effective_content_mask"],
        dtype=bool,
    )
    semantic_source_candidate_mask = np.asarray(
        semantic_reuse_result["semantic_candidate_mask"],
        dtype=bool,
    )
    semantic_source_exclusion_mask = np.asarray(
        semantic_reuse_result["excluded_inside_projection_mask"],
        dtype=bool,
    )
    semantic_source_roof_mask = np.asarray(
        semantic_reuse_result["selected_roof_mask"],
        dtype=bool,
    )
    side_content_extension = np.asarray(
        source_side_evidence.get(
            "content_extension_mask", np.zeros((H, W), dtype=bool),
        ),
        dtype=bool,
    )
    if side_content_extension.shape != (H, W):
        side_content_extension = np.zeros((H, W), dtype=bool)
    # Potential outside content is present only on the Hough inspection
    # canvas.  It is not promoted to retained facade content unless that
    # specific side later selects a validated outside edge.
    side_detection_source_mask = retained_source_mask | side_content_extension
    refinement_info = {
        key: value
        for key, value in semantic_reuse_result.items()
        if key not in {
            "semantic_candidate_mask",
            "hard_occluder_mask",
            "generic_non_target_mask",
            "effective_content_mask",
            "excluded_inside_projection_mask",
            "selected_roof_mask",
        }
    }
    refinement_info.update({
        "accepted_for_reuse": bool(semantic_reuse_result["accepted"]),
        "source_stage": "full_image_prefit_semantics_used_for_global_depth_fit",
        "downstream_stage": "source_crop_then_rectification_and_hough",
        "facade_side_evidence": side_evidence_metadata(
            source_side_evidence
        ),
        "opening_sam3": opening_sam_info,
        "side_content_extension_candidate_pixels": int(
            side_content_extension.sum()
        ),
    })
    _, external_mask_inside_projection = _remove_external_building_pixels(
        projection_mask_full,
        selected_external_building_mask,
    )
    status = (
        "accepted"
        if refinement_info["accepted_for_reuse"]
        else "fitted-projection fallback"
    )
    print(
        f"[{facade_tag}] reused pre-fit semantic facade mask {status} | "
        f"reason={refinement_info.get('reason')} | "
        f"coverage={float(refinement_info.get('candidate_wall_coverage', 0.0)):.3f}"
    )
    alpha_build = side_detection_source_mask.astype(np.uint8) * 255
    external_building_occlusion_info["removed_inside_projection_pixel_count"] = int(
        external_mask_inside_projection.sum()
    )
    rgba_full = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_build)))

    projection_crop_path = Path(
        per_building_out,
        f"{geojson_base}__{facade_tag}__projection_cropped_facade.png",
    )
    rgba_full.save(projection_crop_path)
    if stage_timer is not None:
        stage_timer.record(
            f"{facade_tag} / crop using reused prefit semantic evidence",
            time.perf_counter() - semantic_reuse_t0,
        )

    with _timer_stage(stage_timer, f"{facade_tag} / orthorectify"):
        rgba_for_rectify = np.array(rgba_full, dtype=np.uint8)
        H_pix_to_wall_m = homography_from_4pts(uv_rect.astype(float), rect_m.astype(float))

        a2 = rgba_for_rectify[:, :, 3]
        if a2.max() == 0:
            print(f"[{facade_tag}] Empty alpha before orthorectification - falling back to per-fragment texturing.")
            return {}

        target_m = np.vstack([outline_m, rect_m]).astype(np.float64)

        xmin = float(target_m[:, 0].min()) - MARGIN_METERS
        ymin = float(target_m[:, 1].min()) - MARGIN_METERS
        xmax = float(target_m[:, 0].max()) + MARGIN_METERS
        ymax = float(target_m[:, 1].max()) + MARGIN_METERS

        if FLIP_VERTICAL == "auto":
            flip = choose_orientation_from_poly(rect_m, xmin, ymin, xmax, ymax, PIXELS_PER_METER)
        else:
            flip = bool(FLIP_VERTICAL)

        S_m_to_px = S_meter_to_pixel(xmin, ymin, xmax, ymax, PIXELS_PER_METER, flip=flip)
        H_pix_to_ortho_px = S_m_to_px @ H_pix_to_wall_m
        Wm, Hm = (xmax - xmin), (ymax - ymin)
        out_Wr = max(int(np.ceil(Wm * PIXELS_PER_METER)), 1)
        out_Hr = max(int(np.ceil(Hm * PIXELS_PER_METER)), 1)

        area_r = out_Wr * out_Hr
        if area_r > MAX_ORTHO_PIXELS:
            scale = math.sqrt(MAX_ORTHO_PIXELS / float(area_r))
            ppm2 = PIXELS_PER_METER * scale
            S_m_to_px = S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm2, flip=flip)
            H_pix_to_ortho_px = S_m_to_px @ H_pix_to_wall_m
            out_Wr = max(1, int(np.ceil((xmax - xmin) * ppm2)))
            out_Hr = max(1, int(np.ceil((ymax - ymin) * ppm2)))

        src_bgra = cv2.cvtColor(rgba_for_rectify, cv2.COLOR_RGBA2BGRA)
        ortho_bgra = cv2.warpPerspective(
            src_bgra,
            H_pix_to_ortho_px,
            (out_Wr, out_Hr),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        ortho_rgba = cv2.cvtColor(ortho_bgra, cv2.COLOR_BGRA2RGBA)
        wall_poly_px = apply_H(outline_m, S_m_to_px)
        rect_poly_px = apply_H(rect_m, S_m_to_px)
        rectified_wall_mask = build_wall_region_mask(
            out_Hr, out_Wr, wall_poly_px,
        ) > 0
        rectified_semantic_content_mask = cv2.warpPerspective(
            retained_source_mask.astype(np.uint8) * 255,
            H_pix_to_ortho_px,
            (out_Wr, out_Hr),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        rectified_semantic_candidate_mask = cv2.warpPerspective(
            semantic_source_candidate_mask.astype(np.uint8) * 255,
            H_pix_to_ortho_px,
            (out_Wr, out_Hr),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        rectified_semantic_exclusion_mask = cv2.warpPerspective(
            semantic_source_exclusion_mask.astype(np.uint8) * 255,
            H_pix_to_ortho_px,
            (out_Wr, out_Hr),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        rectified_semantic_roof_mask = cv2.warpPerspective(
            semantic_source_roof_mask.astype(np.uint8) * 255,
            H_pix_to_ortho_px,
            (out_Wr, out_Hr),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        # Keep the narrow, side-specific semantic recovery strip until the
        # accepted side/opening warp has been applied.  Clipping here used to
        # destroy a real wall edge whenever it lay just outside the projected
        # target and inside a visible adjacent-wall projection.
        rectified_side_evidence = warp_side_evidence_to_rectified(
            source_side_evidence,
            H_pix_to_ortho_px,
            (out_Hr, out_Wr),
        )
        rectified_side_detection_mask = cv2.warpPerspective(
            side_detection_source_mask.astype(np.uint8) * 255,
            H_pix_to_ortho_px,
            (out_Wr, out_Hr),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        ortho_rgba[~rectified_side_detection_mask, :3] = 0
        ortho_rgba[:, :, 3] = (
            rectified_side_detection_mask.astype(np.uint8) * 255
        )

    ortho_prefit_overlay_path = Path(per_building_out) / f"{geojson_base}__{facade_tag}__ortho_prefit_overlay.png"
    if SAVE_FACADE_GROUP_DEBUG_PNG or SAVE_ARTIFACT_CONTACT_SHEET:
        save_with_overlay(
            Image.fromarray(ortho_rgba).convert("RGBA"),
            wall_poly_px,
            str(ortho_prefit_overlay_path)
        )

    with _timer_stage(
        stage_timer,
        f"{facade_tag} / Hough adjustment with semantic-mask propagation",
    ):
        (
            ortho_rgba,
            hough_info,
            hough_overlay_path,
            hough_warp_overlay_path,
            hough_band_paths,
            hough_auxiliary_masks,
        ) = _apply_group_hough_adjustment(
            ortho_rgba=ortho_rgba,
            wall_poly_px=wall_poly_px,
            rect_poly_px=rect_poly_px,
            per_building_out=per_building_out,
            geojson_base=geojson_base,
            facade_tag=facade_tag,
            edge_mask_override=None,
            allow_guided_warp=True,
            auxiliary_masks={
                "semantic_content": rectified_semantic_content_mask,
                "semantic_candidate": rectified_semantic_candidate_mask,
                "semantic_exclusion": rectified_semantic_exclusion_mask,
                "semantic_roof": rectified_semantic_roof_mask,
            },
            side_evidence=rectified_side_evidence,
            opening_context={
                "source_rows": opening_sam_rows,
                "source_wall_mask": projection_mask_full,
                "source_exclusion_mask": (
                    semantic_source_exclusion_mask
                    | semantic_source_roof_mask
                    | np.asarray(
                        source_side_evidence.get(
                            "foreground_mask",
                            np.zeros((H, W), dtype=bool),
                        ),
                        dtype=bool,
                    )
                ),
                "source_to_rectified_h": H_pix_to_ortho_px,
                "source_rgba": rgba_for_rectify,
                "source_side_extensions": {
                    str(side): np.asarray(
                        row.get(
                            "candidate_extension_mask",
                            np.zeros((H, W), dtype=bool),
                        ),
                        dtype=bool,
                    )
                    for side, row in dict(
                        source_side_evidence.get("sides") or {}
                    ).items()
                },
                "source_masks": {
                    "semantic_content": retained_source_mask,
                    "semantic_candidate": semantic_source_candidate_mask,
                    "semantic_exclusion": semantic_source_exclusion_mask,
                    "semantic_roof": semantic_source_roof_mask,
                },
            },
        )
    reused_rectified_mask = np.asarray(
        hough_auxiliary_masks["semantic_content"],
        dtype=bool,
    ) & rectified_wall_mask
    reused_rectified_candidate_mask = np.asarray(
        hough_auxiliary_masks["semantic_candidate"],
        dtype=bool,
    ) & rectified_wall_mask
    reused_rectified_exclusion_mask = np.asarray(
        hough_auxiliary_masks["semantic_exclusion"],
        dtype=bool,
    ) & rectified_wall_mask
    reused_rectified_roof_mask = np.asarray(
        hough_auxiliary_masks["semantic_roof"],
        dtype=bool,
    ) & rectified_wall_mask

    roof_structure_result = build_post_hough_roof_structure_removal(
        rectified_wall_mask,
        reused_rectified_roof_mask,
        enabled=bool(globals().get(
            "ENABLE_POST_HOUGH_ROOF_STRUCTURE_REMOVAL",
            True,
        )),
        connection_tolerance_px=int(globals().get(
            "POST_HOUGH_ROOF_CONNECTION_TOLERANCE_PX",
            3,
        )),
        boundary_seed_px=int(globals().get(
            "POST_HOUGH_ROOF_BOUNDARY_SEED_PX",
            2,
        )),
        minimum_divider_component_area_px=int(globals().get(
            "POST_HOUGH_ROOF_MIN_DIVIDER_COMPONENT_AREA_PX",
            32,
        )),
        minimum_partition_area_px=int(globals().get(
            "POST_HOUGH_ROOF_MIN_PARTITION_AREA_PX",
            80,
        )),
        minimum_partition_fraction=float(globals().get(
            "POST_HOUGH_ROOF_MIN_PARTITION_FRACTION",
            0.03,
        )),
    )
    roof_structure_removal_mask = np.asarray(
        roof_structure_result["removal_mask"],
        dtype=bool,
    )
    reused_rectified_mask &= ~roof_structure_removal_mask
    reused_rectified_candidate_mask &= ~roof_structure_removal_mask
    reused_rectified_exclusion_mask |= roof_structure_removal_mask
    roof_structure_info = {
        key: value
        for key, value in roof_structure_result.items()
        if key not in {
            "roof_mask",
            "below_roof_mask",
            "removal_mask",
        }
    }
    refinement_info["post_hough_roof_structure_removal"] = (
        roof_structure_info
    )
    if int(roof_structure_info.get("roof_pixels", 0)) > 0:
        print(
            f"[{facade_tag}] post-Hough roof removal | "
            f"roof components={roof_structure_info['roof_component_count']} | "
            f"dividers={roof_structure_info['divider_component_count']} | "
            f"removed={roof_structure_info['removed_pixels']}px"
        )
    ortho_rgba[~reused_rectified_mask, :3] = 0
    ortho_rgba[:, :, 3] = reused_rectified_mask.astype(np.uint8) * 255
    ortho_rgba[~rectified_wall_mask, :3] = 0
    ortho_rgba[~rectified_wall_mask, 3] = 0

    refinement_info.update({
        "rectified_content_pixels_after_hough": int(
            reused_rectified_mask.sum()
        ),
        "rectified_candidate_pixels_after_hough": int(
            reused_rectified_candidate_mask.sum()
        ),
        "rectified_exclusion_pixels_after_hough": int(
            reused_rectified_exclusion_mask.sum()
        ),
        "rectified_roof_evidence_pixels_after_hough": int(
            reused_rectified_roof_mask.sum()
        ),
        "hough_warp_applied_to_rgb_and_semantic_mask": bool(
            hough_info.get("guided_warp_applied", False)
        ),
        "semantic_mask_interpolation": "nearest",
        "second_facade_segmentation_inference_run": False,
        "opening_segmentation_inference_run": bool(
            opening_sam_info.get("reason") == "completed"
        ),
        "side_content_extension_accepted_pixels": int(
            hough_info.get("accepted_side_extension_pixels", 0)
        ),
        "side_content_extension_accepted_sides": list(
            hough_info.get("accepted_side_extension_sides", [])
        ),
    })
    semantic_reuse_overlay_path = Path(
        per_building_out,
        (
            f"{geojson_base}__{facade_tag}"
            "__reused_prefit_semantic_mask_after_hough.png"
        ),
    )
    if (
        bool(globals().get(
            "SAVE_PREFIT_SEMANTIC_TEXTURE_REUSE_DEBUG",
            True,
        ))
        or SAVE_ARTIFACT_CONTACT_SHEET
    ):
        _save_reused_prefit_semantic_overlay(
            img_rgba=ortho_rgba,
            wall_poly_px=wall_poly_px,
            content_mask=reused_rectified_mask,
            exclusion_mask=(
                rectified_wall_mask & (~reused_rectified_mask)
            ),
            out_path=str(semantic_reuse_overlay_path),
            reuse_info=refinement_info,
        )

    ortho_fit_overlay_path = Path(
        per_building_out,
        f"{geojson_base}__{facade_tag}__ortho_fit_overlay.png",
    )
    ortho_fit_source_mask = reused_rectified_mask.copy()
    with _timer_stage(stage_timer, f"{facade_tag} / ortho fit"):
        if (
            reuse_prefit_semantic_mask_enabled
            and not bool(refinement_info.get("accepted_for_reuse", False))
        ):
            M_ortho_fit = None
            ortho_fit_source_pts = None
            ortho_fit_fitted_pts = None
            ortho_fit_info = {
                "enabled": bool(_ortho_fit_enabled()),
                "applied": False,
                "fit_mode": "reused_prefit_semantic_mask_refinement",
                "reason": (
                    "prefit_semantic_target_not_accepted_"
                    "projection_fallback_unchanged"
                ),
                "source_area_px": int(reused_rectified_mask.sum()),
                "target_area_px": int(rectified_wall_mask.sum()),
            }
        else:
            ortho_rgba, M_ortho_fit, ortho_fit_source_pts, ortho_fit_fitted_pts, ortho_fit_info = _fit_ortho_rgba_alpha_inside_polygon(
                ortho_rgba,
                wall_poly_px,
                source_mask_override=reused_rectified_mask,
                max_scale_delta=float(globals().get(
                    "PREFIT_SEMANTIC_TEXTURE_MAX_SCALE_DELTA",
                    0.08,
                )),
                max_translation_px=float(globals().get(
                    "PREFIT_SEMANTIC_TEXTURE_MAX_TRANSLATION_PX",
                    30.0,
                )),
            )
    if ortho_fit_info.get("applied"):
        print(
            f"   ortho fit applied on {facade_tag} | "
            f"scale={ortho_fit_info['scale']:.4f} | "
            f"min_dist={ortho_fit_info['min_signed_dist_px']:.3f}px | "
            f"center_dist={ortho_fit_info['center_dist']:.4f}"
        )
    else:
        print(f"   ortho fit skipped on {facade_tag}: {ortho_fit_info.get('reason')}")

    ortho_fit_display_mask = ortho_fit_source_mask.copy()
    if ortho_fit_info.get("applied") and M_ortho_fit is not None:
        ortho_fit_display_mask = cv2.warpAffine(
            ortho_fit_source_mask.astype(np.uint8) * 255,
            np.asarray(M_ortho_fit, dtype=np.float32),
            (ortho_rgba.shape[1], ortho_rgba.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
    ortho_fit_display_mask &= rectified_wall_mask
    refinement_info.update({
        "optional_ortho_affine_applied_to_rgb_and_semantic_mask": bool(
            ortho_fit_info.get("applied", False)
        ),
        "mask_transform_chain": [
            "selected_source_full_image",
            "fitted_wall_projection_with_validated_side_extension",
            "perspective_rectification_nearest",
            (
                "opening_aware_shared_homography_nearest_one_pass"
                if bool(
                    (hough_info.get("opening_aware") or {}).get(
                        "applied", False
                    )
                )
                else (
                    "validated_side_hough_remap_nearest"
                    if bool(hough_info.get("guided_warp_applied", False))
                    else "side_and_opening_rectification_not_applied"
                )
            ),
            "clip_to_projected_wall_after_rectification",
            "post_hough_roof_structure_removal",
            (
                "optional_ortho_affine_nearest"
                if ortho_fit_info.get("applied", False)
                else "optional_ortho_affine_not_applied"
            ),
        ],
    })

    if SAVE_FACADE_GROUP_DEBUG_PNG or SAVE_ARTIFACT_CONTACT_SHEET:
        _save_ortho_fit_debug_overlay(
            img_rgba=ortho_rgba,
            wall_poly_px=wall_poly_px,
            source_pts=ortho_fit_source_pts,
            fitted_pts=ortho_fit_fitted_pts,
            out_path=str(ortho_fit_overlay_path),
            fit_info=ortho_fit_info,
            source_mask=ortho_fit_source_mask,
            display_mask=ortho_fit_display_mask,
        )

    lama_valid_content_mask = ortho_fit_display_mask
    refinement_info["used_as_lama_valid_content_mask"] = True
    refinement_info["lama_valid_content_area_px"] = int(
        lama_valid_content_mask.sum()
    )

    if ENABLE_LAMA_FILL:
        lama_mask_path = (
            Path(per_building_out) / f"{geojson_base}__{facade_tag}__ortho_lama_mask.png"
            if LAMA_SAVE_DEBUG_MASK else None
        )
        with _timer_stage(stage_timer, f"{facade_tag} / LaMa fill"):
            ortho_rgba, lama_hole_mask = lama_fill_rectified_wall(
                ortho_rgba=ortho_rgba,
                wall_poly_px=wall_poly_px,
                debug_mask_path=str(lama_mask_path) if lama_mask_path is not None else None,
                valid_content_mask=lama_valid_content_mask,
            )
        filled_px = int((lama_hole_mask > 0).sum())
        if filled_px > 0:
            print(f"   LaMa filled {filled_px} pixels on {facade_tag}")

    texture_apply_t0 = time.perf_counter()
    ortho_rgba_full_debug = ortho_rgba.copy()
    if FIT_CLIP_TO_WALL:
        wall_mask_clip = build_wall_region_mask(
            ortho_rgba.shape[0],
            ortho_rgba.shape[1],
            wall_poly_px
        ) > 0
        ortho_rgba_texture = ortho_rgba.copy()
        ortho_rgba_texture[~wall_mask_clip, :3] = 0
        ortho_rgba_texture[~wall_mask_clip, 3] = 0
    else:
        ortho_rgba_texture = ortho_rgba.copy()

    ortho_rgba_texture = bleed_rgb_into_transparency(
        ortho_rgba_texture,
        radius_px=TEXTURE_TRANSPARENT_EDGE_BLEED_PX,
    )

    out_png_ortho = Path(per_building_out) / f"{geojson_base}__{facade_tag}__ortho.png"
    Image.fromarray(ortho_rgba_texture).save(out_png_ortho)

    ortho_overlay_path = Path(per_building_out) / f"{geojson_base}__{facade_tag}__ortho_overlay.png"
    save_with_overlay(
        Image.fromarray(ortho_rgba_full_debug).convert("RGBA"),
        wall_poly_px,
        str(ortho_overlay_path)
    )

    texture_img = Image.fromarray(ortho_rgba_texture).convert("RGBA")
    textured_uv_px = _apply_facade_texture_to_fragments(
        group_records,
        texture_img,
        frame,
        S_m_to_px,
        out_Wr,
        out_Hr,
        mesh_by_name
    )

    if not textured_uv_px:
        return {}
    if stage_timer is not None:
        stage_timer.record(
            f"{facade_tag} / final texture save+apply",
            time.perf_counter() - texture_apply_t0,
        )

    meta_t0 = time.perf_counter()
    out_json_ortho = Path(per_building_out) / f"{geojson_base}__{facade_tag}__ortho_meta.json"
    group_meta = {
        "type": "rectified_facade_group_texture",
        "version": "5.0-opening-aware-side-evidence",
        "geojson": geojson_base,
        "facade_tag": facade_tag,
        "component_id": int(cid) if cid is not None else -1,
        "loop_id": int(lid) if lid is not None else -1,
        "group_id": int(group_id),
        "source_mode": source_mode,
        "wall_global_indices": [int(r["global_index"]) for r in group_records],
        "wall_loop_indices": [int(r["loop_index"]) for r in group_records],
        "source_sv": {
            "pano_id": rec["pano_id"],
            "pano_lat": float(rec["lat"]),
            "pano_lng": float(rec["lng"]),
            "pano_copyright": str(rec.get("copyright", "")),
            "pano_date": rec.get("date"),
            "imagery_provider": str(rec.get("imagery_provider", "unknown")),
            "search_source": str(rec.get("search_source", "")),
            "all_request_urls": [_mask_key(u) for u in urls_fetched],
        },
        "source_candidates": source_candidate_meta,
        "source_selection": {
            "mode": source_mode,
            "requested_policy": str(source_result.get(
                "source_selection_policy_requested",
                source_selection_policy,
            )),
            "effective_policy": str(source_result.get(
                "source_selection_policy",
                "projected_coverage",
            )),
            "method": str(source_result.get(
                "source_selection_method",
                "maximum_net_target_visibility",
            )),
            "selected_source_index": int(
                source_result.get("selected_source_index", 0)
            ),
            "evaluated_source_count": int(len(source_candidate_meta)),
            "ranking": source_result.get("source_selection_ranking", []),
        },
        "external_building_occlusion": {
            **external_building_occlusion_info,
            "enabled": bool((osm_occlusion_context or {}).get("enabled", False)),
            "osm_context_available": bool(
                (osm_occlusion_context or {}).get("available", False)
            ),
            "osm_context_reason": str(
                (osm_occlusion_context or {}).get("reason", "not_provided")
            ),
            "nearby_building_count": int(
                (osm_occlusion_context or {}).get("nearby_building_count", 0)
            ),
            "external_blocker_count": int(
                (osm_occlusion_context or {}).get("external_blocker_count", 0)
            ),
            "excluded_target_buildings": list(
                (osm_occlusion_context or {}).get("excluded_target_buildings", [])
            ),
        },
        "image_space_wireframe_fit": image_space_wireframe_fit,
        "facade_alignment": facade_alignment_info,
        "parallel_model_depth_boundary_fit": depth_boundary_fit_info,
        "depth_aware_region_fit_using_prefit_semantics": (
            depth_aware_region_fit_info
        ),
        "camera_utm_xyz": [float(v) for v in cam.tolist()],
        "camera_elevation": camera_elevation_info,
        "heading_deg": float(heading),
        "heading_reference": "true_north_google_request",
        "projection_heading_deg": float(projection_heading),
        "projection_heading_reference": "source_crs_grid_north",
        "meridian_convergence_deg": float(meridian_convergence),
        "pitch_deg": float(pitch),
        "fov_deg": float(fov_deg),
        "facade_outline_metric_px": [[float(x), float(y)] for x, y in wall_poly_px.tolist()],
        "rectification": {
            "pixels_per_meter": float(PIXELS_PER_METER),
            "margin_m": float(MARGIN_METERS),
            "bounds_m": {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)},
            "bounds_include_selected_alpha": False,
            "selected_alpha_points_for_bounds": 0,
            "flip_vertical": bool(flip),
            "H_pix_to_wall_m": [[float(v) for v in row] for row in H_pix_to_wall_m.tolist()],
            "S_m_to_px": [[float(v) for v in row] for row in S_m_to_px.tolist()],
        },
        "lr_alpha_gate": lr_alpha_gate_info,
        "projection_first_extraction": {
            "enabled": True,
            "semantic_reuse_enabled": bool(
                reuse_prefit_semantic_mask_enabled
            ),
            "source_alpha": (
                "reused_full_image_sam3_target_plus_side_candidate_canvas"
                if refinement_info.get("accepted_for_reuse", False)
                else "fitted_wall_projection_fallback_plus_side_candidate_canvas"
            ),
            "semantic_source_stage": (
                "selected_source_full_image_before_global_depth_fit"
            ),
            "second_facade_segmentation_inference_run": False,
            "opening_segmentation_inference_run": bool(
                opening_sam_info.get("reason") == "completed"
            ),
            "semantic_mask_can_expand_outside_projection": bool(
                refinement_info.get(
                    "side_content_extension_accepted_pixels", 0
                ) > 0
            ),
            "side_extension_is_clipped_after_joint_warp": True,
            "side_extension_promotion_requires_validated_outside_edge": True,
            "semantic_empty_fallback": (
                "fitted_wall_projection_minus_known_and_osm_occluders"
            ),
            "external_osm_exclusion_applied": bool(
                selected_external_building_mask is not None
            ),
        },
        "prefit_semantic_texture_mask_reuse": refinement_info,
        "facade_side_evidence": side_evidence_metadata(
            source_side_evidence
        ),
        "opening_sam3": opening_sam_info,
        "ortho_fit": {
            **ortho_fit_info,
            "target_polygon_px": [[float(x), float(y)] for x, y in wall_poly_px.tolist()],
            "M_fit_2x3": (
                [[float(v) for v in row] for row in M_ortho_fit.tolist()]
                if M_ortho_fit is not None else None
            ),
        },
        "ortho_hough_line_detection": hough_info,
        "artifacts": {
            "sv_rgb_jpg": sv_jpg_name,
            "wireframe_fit_overlay_png": (
                str(wireframe_fit_overlay_path.name)
                if wireframe_fit_overlay_path.is_file() else None
            ),
            "selected_alignment_overlay_png": (
                str(selected_alignment_overlay_path.name)
                if selected_alignment_overlay_path.is_file() else None
            ),
            "projection_cropped_facade_png": str(projection_crop_path.name),
            "external_building_removal_mask_png": (
                external_removal_mask_path.name
                if external_removal_mask_path is not None
                and external_removal_mask_path.is_file()
                else None
            ),
            "selected_source_external_buildings_removed_png": (
                external_removed_preview_path.name
                if external_removed_preview_path is not None
                and external_removed_preview_path.is_file()
                else None
            ),
            "depth_aware_region_fit_overlay_png": (
                str(depth_aware_region_fit_overlay_path.name)
                if depth_aware_region_fit_overlay_path.is_file() else None
            ),
            "reused_prefit_semantic_mask_overlay_png": (
                str(semantic_reuse_overlay_path.name)
                if semantic_reuse_overlay_path is not None
                and semantic_reuse_overlay_path.is_file()
                else None
            ),
            "guarded_ortho_fit_overlay_png": (
                str(ortho_fit_overlay_path.name)
                if ortho_fit_overlay_path.is_file()
                else None
            ),
            "ortho_prefit_overlay_png": str(ortho_prefit_overlay_path.name),
            "ortho_fit_overlay_png": str(ortho_fit_overlay_path.name),
            "hough_overlay_png": str(Path(hough_overlay_path).name) if hough_overlay_path is not None else None,
            "hough_warp_overlay_png": str(Path(hough_warp_overlay_path).name) if hough_warp_overlay_path is not None else None,
            "opening_aware_overlay_png": (
                (hough_info.get("opening_aware") or {}).get(
                    "overlay_png"
                )
            ),
            "hough_band_pngs": {
                str(k): str(Path(v).name)
                for k, v in hough_band_paths.items()
            },
            "ortho_png": str(out_png_ortho.name),
            "ortho_overlay_png": str(ortho_overlay_path.name),
            "model_depth": depth_artifacts or None,
            "model_depth_boundary_fit": depth_boundary_artifacts or None,
        },
    }
    with open(out_json_ortho, "w", encoding="utf-8") as f:
        json.dump(group_meta, f, ensure_ascii=False, indent=2)
    if stage_timer is not None:
        stage_timer.record(
            f"{facade_tag} / write metadata",
            time.perf_counter() - meta_t0,
        )

    viewer_rows = {}
    for rec_wall in group_records:
        gi = int(rec_wall["global_index"])
        q = rec_wall["wall_quad"]
        viewer_rows[gi] = {
            "geojson": geojson_base,
            "wall_tag": f"c{cid_tag}_l{lid_tag}_w{gi:02d}",
            "facade_group_tag": facade_tag,
            "facade_group_id": int(group_id),
            "facade_group_wall_indices": [int(r["global_index"]) for r in group_records],
            "facade_group_fragment_count": int(len(group_records)),
            "component_id": int(cid) if cid is not None else -1,
            "loop_id": int(lid) if lid is not None else -1,
            "loop_index": int(rec_wall["loop_index"]),
            "global_index": gi,
            "source_mode": source_mode,
            "pano_id": rec["pano_id"],
            "pano_lat": float(rec["lat"]),
            "pano_lng": float(rec["lng"]),
            "pano_copyright": str(rec.get("copyright", "")),
            "pano_date": rec.get("date"),
            "imagery_provider": str(rec.get("imagery_provider", "unknown")),
            "camera_utm_xyz": [float(v) for v in cam.tolist()],
            "camera_elevation": camera_elevation_info,
            "heading_deg": float(heading),
            "heading_reference": "true_north_google_request",
            "projection_heading_deg": float(projection_heading),
            "projection_heading_reference": "source_crs_grid_north",
            "meridian_convergence_deg": float(meridian_convergence),
            "pitch_deg": float(pitch),
            "fov_deg": float(fov_deg),
            "wall_quad_xyz_b1b2t2t1": [[float(a), float(b), float(c)] for a, b, c in q.tolist()],
            "wall_uv_px_in_facade_texture": [[float(x), float(y)] for x, y in textured_uv_px[gi].tolist()],
            "ortho_png": str(out_png_ortho.name),
            "sv_rgb_jpg": sv_jpg_name,
            "model_depth": depth_artifacts or None,
            "facade_alignment": facade_alignment_info,
            "projection_first_extraction": True,
            "prefit_semantic_texture_mask_reuse": refinement_info,
            "model_depth_boundary_fit": depth_boundary_fit_info,
            "model_depth_boundary_fit_artifacts": depth_boundary_artifacts or None,
            "depth_aware_region_fit": depth_aware_region_fit_info,
            "depth_aware_region_fit_overlay_png": (
                str(depth_aware_region_fit_overlay_path.name)
                if depth_aware_region_fit_overlay_path.is_file() else None
            ),
        }

    print(f"   Saved grouped facade texture: {out_png_ortho.name} ({len(viewer_rows)} fragments)")
    return viewer_rows

def process_building(geojson_path: str,
                     out_root: str,
                     geotiff_path: Optional[str] = None,
                      *,
                      device=None,
                      processor=None,
                      sam3_prompt_facade=None,
                      sam3_prompt_facade_refinement=None,
                      sam3_prompt_roof=None):

    if not str(API_KEY).strip() or str(API_KEY).startswith("YOUR_"):
        raise ValueError(
            "Google Street View API key is not configured. Copy "
            "lod2_texture_pipeline/config_local.example.py to "
            "lod2_texture_pipeline/config_local.py and set API_KEY."
        )

    geojson_base = os.path.splitext(os.path.basename(geojson_path))[0]
    per_building_out = os.path.join(out_root, geojson_base)
    ensure_outdir(per_building_out)
    if _wall_group_projection_export_enabled():
        stale_projection_root = _wall_group_projection_export_staging_root(per_building_out)
        if stale_projection_root.exists():
            shutil.rmtree(stale_projection_root)
    stage_timer = _PipelineTimer(geojson_base)
    run_started_at = time.time() - 1.0
    viewer_index = []  # will be saved as viewer_index.json
    all_wall_quads_global = []      # list of (4,3) arrays in global index order
    all_wall_meta_global = []


    with stage_timer.stage("load 3D GeoJSON"):
        gdf, corners, edge_groups, id_to_idx, wall_centers, base_z, surface_faces = load_3d_geojson(geojson_path)
        model_boundary_edges_xyz = _model_boundary_edges_xyz_by_class(
            edge_groups,
            corners,
            id_to_idx,
        )

    # Build loops
    with stage_timer.stage("build edge loops"):
        wall_loops = build_edge_loops_from_gdf(gdf, 'wall')
        base_loops = build_edge_loops_from_gdf(gdf, 'base')
        roof_loops = build_edge_loops_from_gdf(gdf, 'roof')

    if not wall_loops:
        print("No wall loops found. Exiting.")
        stage_timer.finish(per_building_out)
        return

    with stage_timer.stage("validate DGM camera elevation"):
        camera_elevation_resolver = _build_dgm_camera_elevation_resolver(
            geojson_base,
            corners=corners,
            base_edges=edge_groups.get("base", []),
            id_to_idx=id_to_idx,
            base_z=base_z,
        )

    # Base edges for pano search area (use all base lines)
    base_edges_gdf = gdf[gdf['type'] == 'base'].copy()

    # --- pano discovery (one big set around all base lines in this geojson)
    with stage_timer.stage("Street View pano discovery"):
        pano_records = build_search_grid_and_collect_panos(
            list(base_edges_gdf.geometry), transformer, back_tx, API_KEY, offset=GRID_OFFSET_M, n=GRID_N
        )
    if len(pano_records) == 0:
        print("No pano candidates found. Exiting.")
        stage_timer.finish(per_building_out)
        return

    # ---- PREBUILD PLACEHOLDERS ----
    prebuild_t0 = time.perf_counter()
    meshes_named = []
    mesh_by_name = {}

    def _surface_rgba(surface_type: str):
        surface_type = str(surface_type).lower()
        if surface_type == "roof":
            return [220, 220, 220, 255]
        if surface_type == "roof_seam":
            return [200, 125, 45, 255]
        if surface_type == "base":
            return [240, 240, 240, 255]
        return [220, 220, 220, 255]

    explicit_surface_faces = [
        sf for sf in surface_faces
        if str(sf.get("surface_type", "")).lower() in {"roof", "roof_seam", "base"}
    ]
    has_explicit_roof_surfaces = any(
        str(sf.get("surface_type", "")).lower() in {"roof", "roof_seam"}
        for sf in explicit_surface_faces
    )
    has_explicit_base_surfaces = any(
        str(sf.get("surface_type", "")).lower() == "base"
        for sf in explicit_surface_faces
    )

    roof_meshes = []  # (name, mesh, coords) per roof or roof_seam surface
    if has_explicit_roof_surfaces:
        for sidx, sf in enumerate(explicit_surface_faces):
            surface_type = str(sf.get("surface_type", "")).lower()
            if surface_type not in {"roof", "roof_seam"}:
                continue
            mesh, coords = build_trimesh_from_surface_face(
                corners,
                sf,
                flat_rgba=_surface_rgba(surface_type)
            )
            if mesh is None or coords is None:
                continue
            name = f"{surface_type}_{int(sf.get('surface_id', sidx)):02d}"
            meshes_named.append((name, mesh))
            if surface_type in {"roof", "roof_seam"}:
                roof_meshes.append((name, mesh, coords))
    else:
        roof_edges = edge_groups.get("roof", [])
        if roof_edges:
            rc_list, rf_list = triangulate_surface(roof_edges, corners, id_to_idx, split_components=True)
            if rc_list and rf_list:
                for ridx, (rc, rf) in enumerate(zip(rc_list, rf_list)):
                    if rc is None or rf is None:
                        continue
                    m = trimesh.Trimesh(vertices=rc, faces=rf, process=False)
                    m.visual.face_colors = [220, 220, 220, 255]
                    rname = f"roof_{ridx:02d}"
                    meshes_named.append((rname, m))
                    roof_meshes.append((rname, m, rc))


    # Base placeholders per loop
    if has_explicit_base_surfaces:
        for sidx, sf in enumerate(explicit_surface_faces):
            surface_type = str(sf.get("surface_type", "")).lower()
            if surface_type != "base":
                continue
            mesh, coords = build_trimesh_from_surface_face(
                corners,
                sf,
                flat_rgba=_surface_rgba(surface_type)
            )
            if mesh is None:
                continue
            name = f"{surface_type}_{int(sf.get('surface_id', sidx)):02d}"
            meshes_named.append((name, mesh))
    else:
        for blp in base_loops:
            cid, lid = blp['component_id'], blp['loop_id']
            bc, bf = triangulate_surface(blp['edges'], corners, id_to_idx)
            if bc is None or bf is None:
                continue
            name = f"base_c{cid}_l{lid}"
            m = trimesh.Trimesh(vertices=bc, faces=bf, process=False)
            m.visual.face_colors = [240,240,240,255]
            meshes_named.append((name, m))

    # Wall placeholders (white quads) + build wall_quads bundle in the SAME global indexing scheme
    wall_records_by_loop = defaultdict(list)
    global_wall_index = 0
    for loop in wall_loops:
        cid, lid = loop['component_id'], loop['loop_id']
        ring_edges = loop['edges']

        wn, centers, base_segs = compute_wall_normals_from_wall_faces(corners, ring_edges, id_to_idx)

        for k in range(len(ring_edges)):
            i_global = global_wall_index
            global_wall_index += 1  # EXACTLY ONCE per wall face

            (s1, t1) = ring_edges[k]
            (s2, t2) = ring_edges[(k + 1) % len(ring_edges)]

            # Default quad placeholder for robustness (keeps array length aligned)
            wall_quad = np.full((4, 3), np.nan, dtype=np.float64)

            if not any(nid not in id_to_idx for nid in [s1, t1, s2, t2]):
                p1a = corners[id_to_idx[s1]]; p1b = corners[id_to_idx[t1]]
                p2a = corners[id_to_idx[s2]]; p2b = corners[id_to_idx[t2]]

                def by_z(a, b): return (a, b) if a[2] <= b[2] else (b, a)
                b1, t1p = by_z(p1a, p1b)
                b2, t2p = by_z(p2a, p2b)

                wall_quad = np.vstack([b1, b2, t2p, t1p]).astype(np.float64)  # [b1,b2,t2,t1]

            # ---- record for bundle (ALWAYS one entry per wall index) ----
            all_wall_quads_global.append(wall_quad.copy())
            all_wall_meta_global.append({
                "component_id": int(cid) if cid is not None else -1,
                "loop_id": int(lid) if lid is not None else -1,
                "loop_index": int(k),
                "global_index": int(i_global),
            })

            # ---- build placeholder mesh ONLY if quad is finite ----
            mesh_name = None
            if np.isfinite(wall_quad).all():
                name = f"wall_c{cid}_l{lid}_w{i_global:02d}"
                placeholder = _build_wall_mesh_from_verts(
                    wall_quad,
                    outward_normal_xyz=wn[k],
                    uv_px=None, tex_img=None, out_w=None, out_h=None,
                    flat_rgba=(240, 240, 240, 255)
                )
                meshes_named.append((name, placeholder))
                mesh_by_name[name] = placeholder
                mesh_name = name

            wall_records_by_loop[(cid, lid)].append({
                "component_id": cid,
                "loop_id": lid,
                "loop_index": int(k),
                "global_index": int(i_global),
                "edge": (s1, t1),
                "next_edge": (s2, t2),
                "wall_quad": wall_quad,
                "normal": wn[k],
                "center": centers[k],
                "base_seg": base_segs[k],
                "mesh_name": mesh_name,
            })


    stage_timer.record("build placeholder meshes and wall records", time.perf_counter() - prebuild_t0)
    with stage_timer.stage("prepare OSM external-building occlusion context"):
        osm_occlusion_context = _prepare_osm_building_occlusion_context(
            geojson_path,
            base_z,
            camera_elevation_resolver=camera_elevation_resolver,
        )
    with stage_timer.stage("build facade groups + group debug images"):
        facade_group_items = _collect_facade_group_items(wall_records_by_loop)
        _save_facade_group_debug_images(facade_group_items, per_building_out, geojson_base)
        artifact_debug_rows = _facade_group_artifact_debug_rows(facade_group_items, geojson_base)
    wall_group_lookup = {}
    for row in artifact_debug_rows:
        group_info = {
            "facade_group_tag": row.get("facade_group_tag"),
            "facade_group_id": row.get("facade_group_id"),
            "facade_group_wall_indices": row.get("facade_group_wall_indices"),
            "facade_group_fragment_count": row.get("facade_group_fragment_count"),
        }
        wall_group_lookup[int(row["global_index"])] = group_info

    # Panos & model (passed from main; do NOT reload per building)
    if device is None or processor is None or sam3_prompt_facade is None or sam3_prompt_roof is None:
        raise RuntimeError("SAM3 bundle not provided. Pass device/processor/sam3_prompt_facade/sam3_prompt_roof from main().")
    if sam3_prompt_facade_refinement is None:
        sam3_prompt_facade_refinement = globals().get("SAM3_PROMPT_FACADE_REFINEMENT", None)

    grouped_wall_viewer_rows = {}
    grouped_wall_reserved_indices = set()
    configured_source_mode = str(globals().get(
        "FACADE_SOURCE_SELECTION_MODE", "auto",
    )).strip().lower()
    building_source_selection_policy, singleton_only_building = (
        _resolve_facade_source_selection_policy(
            facade_group_items,
            configured_mode=configured_source_mode,
        )
    )
    print(
        "Facade source selection policy: "
        f"{building_source_selection_policy} "
        f"(configured={configured_source_mode}, "
        f"singleton_only={singleton_only_building})."
    )
    grouped_texturing_t0 = time.perf_counter()
    for item in facade_group_items:
        group_id = int(item.get("group_id", -1))
        group_records = item.get("records", [])
        if not group_records:
            continue
        grouped_wall_reserved_indices.update(int(r["global_index"]) for r in group_records)
        try:
            rows = _texture_facade_group(
                group_records=group_records,
                group_id=group_id,
                geojson_base=geojson_base,
                per_building_out=per_building_out,
                base_z=base_z,
                pano_records=pano_records,
                processor=processor,
                mesh_by_name=mesh_by_name,
                meshes_named=meshes_named,
                model_boundary_edges_xyz=model_boundary_edges_xyz,
                osm_occlusion_context=osm_occlusion_context,
                stage_timer=stage_timer,
                source_selection_policy=building_source_selection_policy,
                camera_elevation_resolver=camera_elevation_resolver,
                loop_records=wall_records_by_loop.get(item.get("loop_key"), []),
            )
            grouped_wall_viewer_rows.update(rows)
        except Exception as e:
            idxs = [int(r["global_index"]) for r in group_records]
            print(f"Facade group {group_id} failed for walls {idxs}: {e}")
            traceback.print_exc()
    facade_group_counter = 0
    for (_cid, _lid), records in []:
        facade_groups = _build_facade_groups(records)
        for group_records in facade_groups:
            if not group_records:
                facade_group_counter += 1
                continue
            grouped_wall_reserved_indices.update(int(r["global_index"]) for r in group_records)
            try:
                rows = _texture_facade_group(
                    group_records=group_records,
                    group_id=facade_group_counter,
                    geojson_base=geojson_base,
                    per_building_out=per_building_out,
                    base_z=base_z,
                    pano_records=pano_records,
                    processor=processor,
                    mesh_by_name=mesh_by_name,
                    meshes_named=meshes_named,
                    model_boundary_edges_xyz=model_boundary_edges_xyz,
                    osm_occlusion_context=osm_occlusion_context,
                    stage_timer=stage_timer,
                    source_selection_policy=building_source_selection_policy,
                    camera_elevation_resolver=camera_elevation_resolver,
                    loop_records=wall_records_by_loop.get((_cid, _lid), []),
                )
                grouped_wall_viewer_rows.update(rows)
            except Exception as e:
                idxs = [int(r["global_index"]) for r in group_records]
                print(f"WARNING: Facade group {facade_group_counter} failed for walls {idxs}: {e}")
                traceback.print_exc()
            finally:
                facade_group_counter += 1

    stage_timer.record("grouped facade texturing total", time.perf_counter() - grouped_texturing_t0)
    if grouped_wall_viewer_rows:
        print(f"Grouped facade texturing covered {len(grouped_wall_viewer_rows)} wall fragments.")

    # Reset running global index for consistent naming in outputs
    global_wall_index = 0

    # ======== Per-loop wall processing ========
    legacy_wall_t0 = time.perf_counter()
    for loop in wall_loops:
        cid, lid = loop['component_id'], loop['loop_id']
        ring_edges = loop['edges']

        wall_normals, centers_xyz, base_segs_xy = compute_wall_normals_from_wall_faces(
            corners, ring_edges, id_to_idx
        )
        sel_xy, sel_recs = select_pano_per_wall_using_prism_base(
            ring_edges, wall_normals, corners, id_to_idx, pano_records
        )

        for k, ((s1, t1id), pick_xy, rec, ctr, seg, n_xy) in enumerate(
            zip(ring_edges, sel_xy, sel_recs, centers_xyz, base_segs_xy, [n[:2] for n in wall_normals])
        ):
            i_global = global_wall_index
            global_wall_index += 1

            if i_global in grouped_wall_viewer_rows:
                viewer_index.append(grouped_wall_viewer_rows[i_global])
                continue

            if (
                i_global in grouped_wall_reserved_indices
                and not bool(FACADE_GROUP_ALLOW_PER_WALL_FALLBACK)
            ):
                print(f"[c{cid} l{lid} w{k}] grouped facade reserved this wall - skipping per-wall Street View fallback.")
                continue

            if pick_xy is None or rec is None or ctr is None or seg is None or n_xy is None:
                print(f"[c{cid} l{lid} w{k}] no pano selected - skipping.")
                continue

            px, py = pick_xy
            camera_elevation_decision = camera_elevation_resolver.resolve(
                px,
                py,
                source_label=f"pano {rec.get('pano_id', 'unknown')}",
            )
            cam = np.array(
                [px, py, camera_elevation_decision.camera_z_m],
                dtype=float,
            )

            dx, dy = ctr[0] - px, ctr[1] - py
            dz     = ctr[2] - cam[2]
            heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
            rho     = np.hypot(dx, dy)
            pitch   = np.degrees(np.arctan2(dz, max(rho, 1e-9)))
            fov_deg = solve_fov_deg(np.array([px, py]), heading, seg, n_xy, buffer_m=SIDE_BUFFER_M, safety_margin_deg=FOV_MARGIN_DEG)

            # Wall quad [b1,b2,t2,t1]
            (s1, t1id) = ring_edges[k]
            (s2, t2id) = ring_edges[(k + 1) % len(ring_edges)]
            p1a = corners[id_to_idx[s1]]; p1b = corners[id_to_idx[t1id]]
            p2a = corners[id_to_idx[s2]]; p2b = corners[id_to_idx[t2id]]
            def by_z(a,b): return (a,b) if a[2] <= b[2] else (b,a)
            b1, t1p = by_z(p1a, p1b); b2, t2p = by_z(p2a, p2b)
            wall_quad = np.vstack([b1, b2, t2p, t1p])  # [b1,b2,t2,t1]

            (
                img_rgb,
                uv,
                heading,
                projection_heading,
                pitch,
                fov_deg,
                K,
                R_wc,
                C,
                urls_fetched,
            ) = fetch_single_wall_source(
                rec["pano_id"], cam, wall_quad, heading, pitch, fov_deg, img_size=SV_SIZE
            )
            meridian_convergence = float(
                wrap_delta_deg(heading, projection_heading)
            )

            print(f"[c{cid} l{lid} w{k}] fetched {len(urls_fetched)} Street View image(s):")
            for u in urls_fetched:
                safe_u = u.split("&key=")[0] + "&key=****" if "&key=" in u else u
                print("         URL:", safe_u)

            wall_tag = f"c{cid}_l{lid}_w{i_global:02d}"

            sv_jpg_name = (
                f"sv__{geojson_base}__{wall_tag}"
                f"__pano_{rec['pano_id']}"
                f"__hdg_{int(round(heading))}"
                f"__pit_{int(round(pitch))}"
                f"__fov_{int(round(fov_deg))}"
                + ".jpg"
            )

            sv_jpg_path = os.path.join(per_building_out, sv_jpg_name)
            if SAVE_SV_RGB_PER_WALL or SAVE_ARTIFACT_CONTACT_SHEET:
                img_rgb.convert("RGB").save(sv_jpg_path, quality=95)
            else:
                sv_jpg_name = None  # viewer_index will reflect that


            raw_overlay_path = os.path.join(
                per_building_out,
                name_for("raw_overlay", base=geojson_base, wall=i_global, rec=rec,
                        heading=heading, pitch=pitch, fov=fov_deg)
            )
            if SAVE_RAW_OVERLAY_PNG or SAVE_ARTIFACT_CONTACT_SHEET:
                save_overlay_matplotlib(
                    img_rgb, uv, raw_overlay_path,
                    title=f"Wall {wall_tag} - heading {heading:.1f} deg, pitch {pitch:.1f} deg, fov {fov_deg:.1f} deg"
                )

            # Choose/define the SV jpg filename (make it unambiguous)
            if SAVE_SV_RGB_PER_WALL or SAVE_ARTIFACT_CONTACT_SHEET:
                sv_jpg_name = (
                    f"sv__{geojson_base}__c{cid}_l{lid}_w{i_global:02d}"
                    f"__pano_{rec['pano_id']}__hdg_{int(round(heading))}"
                    f"__pit_{int(round(pitch))}__fov_{int(round(fov_deg))}"
                    + ".jpg"
                )
            else:
                sv_jpg_name = None  

            depth_artifacts = {}
            if bool(globals().get("SAVE_MODEL_DEPTH_MAPS", True)):
                try:
                    source_size = (int(round(float(K[0, 2]) * 2.0)), int(round(float(K[1, 2]) * 2.0)))
                    depth_artifacts = _save_model_depth_map_artifacts(
                        per_building_out=per_building_out,
                        prefix_name=f"{geojson_base}__{wall_tag}__model_depth",
                        meshes_named=meshes_named,
                        K=K,
                        R_wc=R_wc,
                        C=C,
                        source_image_size=source_size,
                        output_image_size=img_rgb.size,
                        image_to_output_H=None,
                        camera_metadata={
                            "mode": "wall_camera",
                            "camera_utm_xyz": [float(v) for v in np.asarray(C, dtype=np.float64).tolist()],
                            "camera_elevation": camera_elevation_decision.as_dict(),
                            "heading_deg": float(heading),
                            "heading_reference": "true_north_google_request",
                            "projection_heading_deg": float(projection_heading),
                            "projection_heading_reference": "source_crs_grid_north",
                            "meridian_convergence_deg": float(meridian_convergence),
                            "pitch_deg": float(pitch),
                            "fov_deg": float(fov_deg),
                        },
                    )
                except Exception as exc:
                    print(f"[c{cid} l{lid} w{k}] model depth map failed: {exc}")

            group_info = wall_group_lookup.get(i_global, {})
            viewer_index.append({
                "geojson": geojson_base,
                "wall_tag": f"c{cid}_l{lid}_w{i_global:02d}",
                "facade_group_tag": group_info.get("facade_group_tag"),
                "facade_group_id": group_info.get("facade_group_id"),
                "facade_group_wall_indices": group_info.get("facade_group_wall_indices"),
                "facade_group_fragment_count": group_info.get("facade_group_fragment_count"),
                "component_id": int(cid) if cid is not None else -1,
                "loop_id": int(lid) if lid is not None else -1,
                "loop_index": int(k),
                "global_index": int(i_global),

                "pano_id": rec["pano_id"],
                "pano_lat": float(rec["lat"]),
                "pano_lng": float(rec["lng"]),
                "pano_copyright": str(rec.get("copyright", "")),
                "pano_date": rec.get("date"),
                "imagery_provider": str(rec.get("imagery_provider", "unknown")),
                "camera_utm_xyz": [float(cam[0]), float(cam[1]), float(cam[2])],
                "camera_elevation": camera_elevation_decision.as_dict(),
                "heading_deg": float(heading),
                "heading_reference": "true_north_google_request",
                "projection_heading_deg": float(projection_heading),
                "projection_heading_reference": "source_crs_grid_north",
                "meridian_convergence_deg": float(meridian_convergence),
                "pitch_deg": float(pitch),
                "fov_deg": float(fov_deg),
                "wall_quad_xyz_b1b2t2t1": [[float(a), float(b), float(c)] for a,b,c in wall_quad.tolist()],
                "sv_rgb_jpg": sv_jpg_name,
                "model_depth": depth_artifacts or None,
            })


            lr_rgba, band_poly, band_bbox = build_lr_band_rgba(img_rgb, uv, LR_BAND_BUFFER_PX)
            if lr_rgba is None:
                print(f"[c{cid} l{lid} w{k}] LR-band failed - skip.")
                continue
            lr_overlay_path = os.path.join(
                per_building_out,
                name_for("lr_band_overlay", base=geojson_base, wall=i_global, rec=rec,
                        heading=heading, pitch=pitch, fov=fov_deg)
            )
            if SAVE_LR_OVERLAY_PNG or SAVE_ARTIFACT_CONTACT_SHEET:
                save_with_overlay(lr_rgba, uv, lr_overlay_path)

            W, H = lr_rgba.size
            r,g,b,a0 = lr_rgba.split()
            alpha_np = np.array(a0, dtype=np.uint8)
            if CROP_TO_ALPHA_BBOX and (alpha_np.min() < 255):
                L,Tp,R2,B2 = _lr_model_crop_bbox(alpha_np, uv, W, H)
                img_for_model = lr_rgba.crop((L,Tp,R2,B2)).convert("RGB")
                off_x, off_y  = L, Tp
                out_Wm2f, out_Hm2f  = R2-L, B2-Tp
            else:
                img_for_model = lr_rgba.convert("RGB")
                off_x = off_y = 0
                out_Wm2f, out_Hm2f  = W, H

            # img_for_model is PIL RGB (cropped or not)
            # We need a boolean mask in that same (out_Hm2f, out_Wm2f) space.

            def _extract_mask_stack(out_obj, H, W):
                masks = None
                if isinstance(out_obj, dict):
                    masks = out_obj.get("masks", out_obj.get("mask", out_obj.get("pred_masks", None)))
                else:
                    masks = getattr(out_obj, "masks", None)

                if masks is None:
                    return np.zeros((0, H, W), dtype=bool)

                if torch.is_tensor(masks):
                    m = masks.detach().float().cpu().numpy()
                else:
                    m = np.asarray(masks)

                # (N,1,H,W) -> (N,H,W)
                if m.ndim == 4 and m.shape[1] == 1:
                    m = m[:, 0]

                # (H,W) -> (1,H,W)
                if m.ndim == 2:
                    m = m[None, ...]

                if m.ndim != 3:
                    return np.zeros((0, H, W), dtype=bool)

                stack = (m > 0.5)

                # keep only non-empty masks
                keep = [stack[i] for i in range(stack.shape[0]) if stack[i].any()]
                if len(keep) == 0:
                    return np.zeros((0, H, W), dtype=bool)

                return np.stack(keep, axis=0)


            def _stack_union(mask_stack, H, W):
                if mask_stack.shape[0] == 0:
                    return np.zeros((H, W), dtype=bool)
                return mask_stack.any(axis=0)


            def _polygon_to_mask(H, W, poly_xy):
                mask = np.zeros((H, W), dtype=np.uint8)
                poly = np.round(poly_xy).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], 255)
                return mask > 0
            
            with torch.no_grad():
                state = processor.set_image(img_for_model)  # caches image embeddings ONCE

                # 1) facade / wall as separate instances
                out_facade = processor.set_text_prompt(state=state, prompt=sam3_prompt_facade)
                facade_stack = _extract_mask_stack(out_facade, out_Hm2f, out_Wm2f)

                # 2) roof can remain unioned
                out_roof = processor.set_text_prompt(state=state, prompt=sam3_prompt_roof)
                roof_stack = _extract_mask_stack(out_roof, out_Hm2f, out_Wm2f)
                roof_mask = _stack_union(roof_stack, out_Hm2f, out_Wm2f)

                # Optional: dilate roof a bit before subtraction
                if ROOF_SUBTRACT_DILATE_PX and ROOF_SUBTRACT_DILATE_PX > 0:
                    k = int(ROOF_SUBTRACT_DILATE_PX)
                    kernel = np.ones((2*k+1, 2*k+1), dtype=np.uint8)
                    roof_mask_u8 = (roof_mask.astype(np.uint8) * 255)
                    roof_mask_u8 = cv2.dilate(roof_mask_u8, kernel, iterations=1)
                    roof_mask = roof_mask_u8 > 0

                # wall polygon in the same cropped model frame
                uv_wall_model = np.array([
                    [uv[0,0] - off_x, uv[0,1] - off_y],  # b1
                    [uv[1,0] - off_x, uv[1,1] - off_y],  # b2
                    [uv[2,0] - off_x, uv[2,1] - off_y],  # t2
                    [uv[3,0] - off_x, uv[3,1] - off_y],  # t1
                ], dtype=float)

                building_mask, selected_idxs, facade_scores, debug_facade_stack, refinement_info = (
                    _select_facade_mask_with_optional_refinement(
                        processor=processor,
                        state=state,
                        facade_stack=facade_stack,
                        roof_mask=roof_mask,
                        wall_poly_xy=uv_wall_model,
                        H=out_Hm2f,
                        W=out_Wm2f,
                        primary_prompt=sam3_prompt_facade,
                        refinement_prompt=sam3_prompt_facade_refinement,
                    )
                )

                if refinement_info.get("accepted"):
                    print(
                        f"[c{cid} l{lid} w{k}] SAM3 refinement accepted | "
                        f"primary={sam3_prompt_facade!r} -> refine={sam3_prompt_facade_refinement!r} | "
                        f"selected: {selected_idxs} | wall_px={refinement_info.get('refinement_wall_pixels')}"
                    )
                elif refinement_info.get("enabled"):
                    print(
                        f"[c{cid} l{lid} w{k}] SAM3 refinement skipped ({refinement_info.get('reason')}) | "
                        f"using primary prompt {sam3_prompt_facade!r}"
                    )
                print(f"[c{cid} l{lid} w{k}] SAM3 selected instances: {debug_facade_stack.shape[0]} | selected: {selected_idxs}")
                for row in facade_scores:
                    i, score, area, inter, outside, center_dist = row
                    print(f"    facade[{i}] score={score:.4f} area={area} inter={inter} outside={outside} center_dist={center_dist:.4f}")

            sam3_instances_overlay_path = Path(per_building_out) / name_for(
                "sam3_instances_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )

            save_sam3_instance_debug_overlay(
                base_img_pil=img_for_model,
                facade_stack=debug_facade_stack,
                roof_mask=roof_mask,
                selected_idx=selected_idxs,
                facade_scores=facade_scores,
                out_path=str(sam3_instances_overlay_path)
            )

            pred_full_raw = np.zeros((H, W), dtype=bool)
            pred_full_raw[off_y:off_y+out_Hm2f, off_x:off_x+out_Wm2f] = building_mask
            pred_full_raw, lr_alpha_gate_info = _apply_lr_alpha_gate_to_selected_mask(
                pred_full_raw,
                alpha_np,
                uv_wall_final,
            )
            if lr_alpha_gate_info.get("rescued_selected_px", 0) > 0:
                print(
                    f"[c{cid} l{lid} w{k}] LR alpha gate rescued "
                    f"{lr_alpha_gate_info['rescued_selected_px']} selected facade pixels"
                )

            # clean the selected full-frame segmentation BEFORE quad fitting
            pred_full_clean = clean_selected_mask(pred_full_raw)
            alpha_build = (pred_full_clean.astype(np.uint8) * 255)

            rgba_full = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_build)))

            # full post-SAM3 frame in ORIGINAL perspective image coordinates
            rgba_final = rgba_full

            uv_wall_final = np.array([
                [uv[0,0], uv[0,1]],  # b1
                [uv[1,0], uv[1,1]],  # b2
                [uv[2,0], uv[2,1]],  # t2
                [uv[3,0], uv[3,1]],  # t1
            ], dtype=float)

            # raw overlay: still useful to inspect the selected segmentation before perspective quad fit
            sam3_overlay_path = Path(per_building_out) / name_for(
                "sam3_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            save_with_overlay(rgba_final, uv_wall_final, str(sam3_overlay_path))
            rgba_for_rectify = np.array(rgba_final, dtype=np.uint8)

            # -----------------------------------------------------------------
            # Orthorectify the current segmentation into wall-plane ortho space
            # -----------------------------------------------------------------
            dst_m, metric_meta = wall_metric_target_from_corners(
                wall_quad[0], wall_quad[1], wall_quad[2], wall_quad[3]
            )
            H_pix_to_wall_m = homography_from_4pts(uv_wall_final.astype(float), dst_m.astype(float))

            a2 = rgba_for_rectify[:, :, 3]
            if a2.max() == 0:
                print(f"[c{cid} l{lid} w{k}] Empty alpha before orthorectification; skip.")
                continue

            ys, xs = np.where(a2 > 0)
            contour_px = np.stack([xs, ys], axis=1).astype(np.float64)
            contour_m = apply_H(contour_px, H_pix_to_wall_m)

            xmin, ymin = contour_m.min(axis=0)
            xmax, ymax = contour_m.max(axis=0)

            xmin = min(xmin, dst_m[:, 0].min()) - MARGIN_METERS
            ymin = min(ymin, dst_m[:, 1].min()) - MARGIN_METERS
            xmax = max(xmax, dst_m[:, 0].max()) + MARGIN_METERS
            ymax = max(ymax, dst_m[:, 1].max()) + MARGIN_METERS

            if FLIP_VERTICAL == "auto":
                flip = choose_orientation_from_poly(dst_m, xmin, ymin, xmax, ymax, PIXELS_PER_METER)
            else:
                flip = bool(FLIP_VERTICAL)

            S_m_to_px = S_meter_to_pixel(xmin, ymin, xmax, ymax, PIXELS_PER_METER, flip=flip)
            H_pix_to_ortho_px = S_m_to_px @ H_pix_to_wall_m
            Wm, Hm = (xmax-xmin), (ymax-ymin)
            out_Wr = max(int(np.ceil(Wm*PIXELS_PER_METER)), 1)
            out_Hr = max(int(np.ceil(Hm*PIXELS_PER_METER)), 1)

            area_r = out_Wr * out_Hr
            if area_r > MAX_ORTHO_PIXELS:
                scale = math.sqrt(MAX_ORTHO_PIXELS / float(area_r))
                ppm2  = PIXELS_PER_METER * scale
                S_m_to_px = S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm2, flip=flip)
                H_pix_to_ortho_px = S_m_to_px @ H_pix_to_wall_m
                out_Wr = max(1, int(np.ceil((xmax - xmin) * ppm2)))
                out_Hr = max(1, int(np.ceil((ymax - ymin) * ppm2)))

            src_rgba_np = np.array(rgba_for_rectify, dtype=np.uint8)
            src_bgra = cv2.cvtColor(src_rgba_np, cv2.COLOR_RGBA2BGRA)
            ortho_bgra = cv2.warpPerspective(
                src_bgra,
                H_pix_to_ortho_px,
                (out_Wr, out_Hr),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            ortho_rgba = cv2.cvtColor(ortho_bgra, cv2.COLOR_BGRA2RGBA)

            wall_poly_px = apply_H(dst_m, S_m_to_px)

            # -----------------------------------------------------------------
            # Unified ortho fit in wall-plane pixel space. A quad wall is the
            # four-point case of the same polygon fitting framework.
            # -----------------------------------------------------------------
            ortho_fit_source_mask = (ortho_rgba[:, :, 3] > 0).copy()
            with _timer_stage(stage_timer, f"wall {i_global:02d} / ortho fit"):
                ortho_rgba, M_ortho_fit, ortho_fit_source_pts, ortho_fit_fitted_pts, ortho_fit_info = _fit_ortho_rgba_alpha_inside_polygon(
                    ortho_rgba,
                    wall_poly_px
                )

            if ortho_fit_info.get("applied"):
                print(
                    f"   ortho fit applied | "
                    f"scale={ortho_fit_info['scale']:.4f} | "
                    f"min_dist={ortho_fit_info['min_signed_dist_px']:.3f}px | "
                    f"center_dist={ortho_fit_info['center_dist']:.4f}"
                )
            else:
                print(f"   ortho fit skipped: {ortho_fit_info.get('reason')}")


            # -----------------------------------------------------------------
            # Hough-based straight-line detection on the ALREADY scaled ortho result
            # -----------------------------------------------------------------
            ortho_alpha_mask_after_fit = ortho_rgba[:, :, 3] > 0

            wall_mask_bool = build_wall_region_mask(
                ortho_rgba.shape[0],
                ortho_rgba.shape[1],
                wall_poly_px
            ) > 0

            hough_edge_map_u8 = build_edge_map_for_hough(
                ortho_rgba[:, :, :3],
                ortho_alpha_mask_after_fit
            )

            hough_lines = []
            hough_left_line = None
            hough_right_line = None
            hough_top_line = None
            hough_left_info = {}
            hough_right_info = {}
            hough_top_info = {}
            hough_total_segments = 0
            hough_warp_overlay_path = None
            hough_guided_warp_applied = False
            ortho_rgba_before_hough_warp = ortho_rgba.copy()

            if ENABLE_ORTHO_HOUGH_DEBUG:
                hough_lines = detect_hough_segments(
                    hough_edge_map_u8,
                    roi_mask=(wall_mask_bool.astype(np.uint8) * 255)
                )
                hough_total_segments = len(hough_lines)

                # wall_poly_px order is [b1, b2, t2, t1]
                left_p0, left_p1 = wall_poly_px[3], wall_poly_px[0]
                right_p0, right_p1 = wall_poly_px[1], wall_poly_px[2]
                top_p0, top_p1 = wall_poly_px[2], wall_poly_px[3]

                left_band_u8 = build_line_search_band(
                    ortho_rgba.shape[0], ortho_rgba.shape[1],
                    left_p0, left_p1, wall_mask_bool, HOUGH_SEARCH_BAND_PX
                )
                right_band_u8 = build_line_search_band(
                    ortho_rgba.shape[0], ortho_rgba.shape[1],
                    right_p0, right_p1, wall_mask_bool, HOUGH_SEARCH_BAND_PX
                )
                top_band_u8 = build_line_search_band(
                    ortho_rgba.shape[0], ortho_rgba.shape[1],
                    top_p0, top_p1, wall_mask_bool, HOUGH_SEARCH_BAND_PX
                )

                hough_left_line, hough_left_info = select_best_hough_line_for_target(
                    hough_lines, left_p0, left_p1, left_band_u8, hough_edge_map_u8,
                    min_length_px=HOUGH_MIN_LENGTH_PX,
                    angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG
                )
                hough_right_line, hough_right_info = select_best_hough_line_for_target(
                    hough_lines, right_p0, right_p1, right_band_u8, hough_edge_map_u8,
                    min_length_px=HOUGH_MIN_LENGTH_PX,
                    angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG
                )
                hough_top_line, hough_top_info = select_best_hough_line_for_target(
                    hough_lines, top_p0, top_p1, top_band_u8, hough_edge_map_u8,
                    min_length_px=HOUGH_MIN_LENGTH_PX,
                    angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG
                )

                print(f"   Hough total segments: {hough_total_segments}")
                print(f"   Hough left:  {hough_left_info}")
                print(f"   Hough right: {hough_right_info}")
                print(f"   Hough top:   {hough_top_info}")

                # -----------------------------------------------------------------
                # Hough-guided ortho warp:
                # align only selected left/right side lines to projected wall lines
                # -----------------------------------------------------------------
                if ENABLE_HOUGH_GUIDED_WARP:
                    detected_side_count = int(hough_left_line is not None) + int(
                        hough_right_line is not None
                    )
                    single_side_allowed = bool(globals().get(
                        "ENABLE_HOUGH_SINGLE_SIDE_WARP", True,
                    ))
                    if (
                        detected_side_count == 2
                        or (single_side_allowed and detected_side_count == 1)
                    ):
                        # projected wall lines from wall_poly_px order [b1, b2, t2, t1]
                        proj_left_line = np.vstack([wall_poly_px[3], wall_poly_px[0]]).astype(np.float64)
                        proj_right_line = np.vstack([wall_poly_px[1], wall_poly_px[2]]).astype(np.float64)
                        ortho_rgba = apply_hough_guided_ortho_warp(
                            ortho_rgba=ortho_rgba,
                            sel_left_line=(
                                hough_left_line.astype(np.float64)
                                if hough_left_line is not None else None
                            ),
                            sel_right_line=(
                                hough_right_line.astype(np.float64)
                                if hough_right_line is not None else None
                            ),
                            sel_top_line=None,
                            proj_left_line=proj_left_line,
                            proj_right_line=proj_right_line,
                            proj_top_line=None
                        )
                        hough_guided_warp_applied = True
                        detected_sides = [
                            side
                            for side, line in (
                                ("left", hough_left_line),
                                ("right", hough_right_line),
                            )
                            if line is not None
                        ]
                        print(
                            "   Hough-guided ortho warp applied "
                            f"using {detected_sides}"
                        )
                    else:
                        print("   Hough-guided ortho warp skipped (no usable side line)")

                    if SAVE_HOUGH_WARP_DEBUG and hough_guided_warp_applied:
                        hough_warp_overlay_path = Path(per_building_out) / name_for(
                            "hough_warp_overlay", base=geojson_base, wall=i_global, rec=rec,
                            heading=heading, pitch=pitch, fov=fov_deg
                        )
                        save_hough_warp_overlay(
                            img_pil=Image.fromarray(ortho_rgba).convert("RGBA"),
                            wall_quad_xy=wall_poly_px,
                            out_path=str(hough_warp_overlay_path)
                        )

                # freeze the post-Hough result for debug overlays
                ortho_rgba_before_hough_warp = ortho_rgba.copy()

                if HOUGH_SAVE_BAND_MASKS:
                    Image.fromarray(left_band_u8 * 255).save(
                        Path(per_building_out) / f"{geojson_base}_wall{i_global:02d}_hough_left_band.png"
                    )
                    Image.fromarray(right_band_u8 * 255).save(
                        Path(per_building_out) / f"{geojson_base}_wall{i_global:02d}_hough_right_band.png"
                    )
                    Image.fromarray(top_band_u8 * 255).save(
                        Path(per_building_out) / f"{geojson_base}_wall{i_global:02d}_hough_top_band.png"
                    )

            ortho_fit_overlay_path = Path(per_building_out) / name_for(
                "ortho_fit_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            _save_ortho_fit_debug_overlay(
                img_rgba=ortho_rgba_before_hough_warp,
                wall_poly_px=wall_poly_px,
                source_pts=ortho_fit_source_pts,
                fitted_pts=ortho_fit_fitted_pts,
                out_path=str(ortho_fit_overlay_path),
                fit_info=ortho_fit_info,
                source_mask=ortho_fit_source_mask,
            )

            hough_overlay_path = Path(per_building_out) / name_for(
                "hough_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            save_hough_all_lines_overlay(
                img_pil=Image.fromarray(ortho_rgba_before_hough_warp).convert("RGBA"),
                wall_quad_xy=wall_poly_px,
                all_lines=hough_lines,
                selected_left=hough_left_line,
                selected_right=hough_right_line,
                selected_top=None,
                out_path=str(hough_overlay_path),
            )

            out_png_ortho = Path(per_building_out) / name_for(
                "ortho_png", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )

            # 2) then fill the remaining uncovered wall area
            if ENABLE_LAMA_FILL:
                lama_mask_path = out_png_ortho.with_name(out_png_ortho.stem + "_lama_mask.png") if LAMA_SAVE_DEBUG_MASK else None

                ortho_rgba, lama_hole_mask = lama_fill_rectified_wall(
                    ortho_rgba=ortho_rgba,
                    wall_poly_px=wall_poly_px,
                    debug_mask_path=str(lama_mask_path) if lama_mask_path is not None else None
                )

                filled_px = int((lama_hole_mask > 0).sum())
                if filled_px > 0:
                    print(f"   LaMa filled {filled_px} pixels on wall {i_global:02d}")

            # keep a FULL version for debug overlay (shows outside-the-wall parts too)
            ortho_rgba_full_debug = ortho_rgba.copy()

            # make a clipped version only for the actual baked texture if requested
            if FIT_CLIP_TO_WALL:
                wall_mask_clip = build_wall_region_mask(
                    ortho_rgba.shape[0], ortho_rgba.shape[1], wall_poly_px
                ) > 0
                ortho_rgba_texture = ortho_rgba.copy()
                ortho_rgba_texture[~wall_mask_clip, :3] = 0
                ortho_rgba_texture[~wall_mask_clip, 3] = 0
            else:
                ortho_rgba_texture = ortho_rgba.copy()

            ortho_rgba_texture = bleed_rgb_into_transparency(
                ortho_rgba_texture,
                radius_px=TEXTURE_TRANSPARENT_EDGE_BLEED_PX,
            )

            Image.fromarray(ortho_rgba_texture).save(out_png_ortho)

            ortho_overlay_path = Path(per_building_out) / name_for(
                "ortho_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            save_with_overlay(
                Image.fromarray(ortho_rgba_full_debug).convert("RGBA"),
                wall_poly_px,
                str(ortho_overlay_path)
            )

            W_img, H_img = img_rgb.size
            fx = fy = (W_img / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
            cx, cy = W_img / 2.0, H_img / 2.0

            quad_xyz_re = np.vstack([wall_quad[3], wall_quad[2], wall_quad[1], wall_quad[0]])  # t1,t2,b2,b1
            out_json_ortho = Path(per_building_out) / name_for(
                "ortho_meta", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg)
            consolidated = {
                "type": "rectified_wall_texture",
                "version": "1.5-per-loop",
                "cropped_image": {
                    "mode": "rgba_band_in_memory",
                    "uv_quad_px_order": "b1,b2,t2,t1",
                    "uv_quad_px_raw": [[float(u), float(v)] for (u, v) in uv.tolist()],
                    "band_polygon_px": band_poly,
                    "band_bbox_px": list(band_bbox) if band_bbox else None,
                },
                "intrinsics": {
                    "fx": float(fx), "fy": float(fy),
                    "cx": float(cx), "cy": float(cy),
                    "fov_deg": float(fov_deg),
                    "image_size_px": [int(W_img), int(H_img)]
                },
                "extrinsics": {
                    "camera_utm_xyz": [float(cam[0]), float(cam[1]), float(cam[2])],
                    "camera_elevation": camera_elevation_decision.as_dict(),
                    "heading_deg": float(heading),
                    "heading_reference": "true_north_google_request",
                    "projection_heading_deg": float(projection_heading),
                    "projection_heading_reference": "source_crs_grid_north",
                    "meridian_convergence_deg": float(meridian_convergence),
                    "pitch_deg": float(pitch),
                    "world_up": [0.0, 0.0, 1.0]
                },
                "wall_identity": {"component_id": int(cid) if cid is not None else -1,
                                  "loop_id": int(lid) if lid is not None else -1,
                                  "loop_index": int(k),
                                  "global_index": int(i_global)},
                "wall_geometry": {
                    "quad_xyz_order": ",".join(OUR_ORDER),
                    "quad_xyz": [[float(a),float(b),float(c)] for a,b,c in quad_xyz_re],
                    "center_xyz": [float(ctr[0]), float(ctr[1]), float(ctr[2])],
                    "metric_target": {
                        "order": "b1,b2,t2,t1",
                        "dst_wall_m": [[float(x), float(y)] for x,y in dst_m.tolist()],
                        **metric_meta
                    }
                },
                "rectification": {
                    "pixels_per_meter": float(PIXELS_PER_METER),
                    "margin_m": float(MARGIN_METERS),
                    "bounds_m": {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)},
                    "flip_vertical": bool(flip),
                    "homography_chain": {
                        "H_pix_to_wall_m": [[float(v) for v in row] for row in H_pix_to_wall_m.tolist()],
                        "S_m_to_px": [[float(v) for v in row] for row in S_m_to_px.tolist()],
                        "H_pix_to_ortho_px": [[float(v) for v in row] for row in H_pix_to_ortho_px.tolist()]
                    }
                },
                "artifacts": {
                    "sam3_alpha_overlay_png": str(Path(sam3_overlay_path).name),
                    "ortho_fit_overlay_png": str(Path(ortho_fit_overlay_path).name),
                    "hough_overlay_png": str(Path(hough_overlay_path).name),
                    "hough_warp_overlay_png": str(Path(hough_warp_overlay_path).name) if hough_warp_overlay_path is not None else None,
                    "ortho_png": str(Path(out_png_ortho).name),
                    "ortho_overlay_png": str(Path(ortho_overlay_path).name),
                    "model_depth": depth_artifacts or None,
                },
                "source_sv": {
                    "pano_id": rec["pano_id"],
                    "street_view_url": f"https://maps.googleapis.com/maps/api/streetview?pano={rec['pano_id']}&size={SV_SIZE}&heading={heading:.4f}&pitch={pitch:.4f}&fov={fov_deg:.4f}&key=****",
                    "all_request_urls": [ _mask_key(u) for u in urls_fetched ],
                    "pano_lat": float(rec["lat"]),
                    "pano_lng": float(rec["lng"]),
                    "pano_copyright": str(rec.get("copyright", "")),
                    "pano_date": rec.get("date"),
                    "imagery_provider": str(rec.get("imagery_provider", "unknown")),
                    "search_source": str(rec.get("search_source", ""))
                },
                "notes": (
                    "Per-loop processing uses one native Street View image; "
                    "framing may issue one wider or recentered request."
                ),
                "lr_alpha_gate": lr_alpha_gate_info,
                "sam3_facade_refinement": refinement_info,
                "post_rectification_fit": {
                    **ortho_fit_info,
                    "target_polygon_px": [[float(x), float(y)] for x, y in wall_poly_px.tolist()],
                    "M_fit_2x3": (
                        [[float(v) for v in row] for row in M_ortho_fit.tolist()]
                        if M_ortho_fit is not None else None
                    ),
                },
                "ortho_hough_line_detection": {
                    "enabled": bool(ENABLE_ORTHO_HOUGH_DEBUG),
                    "method": "canny_plus_houghlinesp_on_scaled_ortho_result",
                    "total_segments_detected": int(hough_total_segments),
                    "config": {
                        "search_band_px": int(HOUGH_SEARCH_BAND_PX),
                        "min_length_px": float(HOUGH_MIN_LENGTH_PX),
                        "max_gap_px": float(HOUGH_MAX_GAP_PX),
                        "angle_thresh_deg": float(HOUGH_ANGLE_THRESH_DEG),
                        "canny_low": int(HOUGH_CANNY_LOW),
                        "canny_high": int(HOUGH_CANNY_HIGH),
                        "canny_dilate_px": int(HOUGH_CANNY_DILATE_PX),
                        "use_clahe": bool(HOUGH_USE_CLAHE),
                    },
                    "left_line_px": (
                        [[float(x), float(y)] for x, y in hough_left_line.tolist()]
                        if hough_left_line is not None else None
                    ),
                    "right_line_px": (
                        [[float(x), float(y)] for x, y in hough_right_line.tolist()]
                        if hough_right_line is not None else None
                    ),
                    "top_line_px": (
                        [[float(x), float(y)] for x, y in hough_top_line.tolist()]
                        if hough_top_line is not None else None
                    ),
                    "left_info": hough_left_info,
                    "right_info": hough_right_info,
                    "top_info": hough_top_info,
                },
            }
            with open(out_json_ortho, "w", encoding="utf-8") as f:
                json.dump(consolidated, f, ensure_ascii=False, indent=2)

            print(f"   Saved overlays and texture: {Path(out_png_ortho).name}")

            # Update wall mesh with the rectified texture
            name = f"wall_c{cid}_l{lid}_w{i_global:02d}"
            uvs_px = np.vstack([  # [b1,b2,t2,t1]
                wall_poly_px[0],
                wall_poly_px[1],
                wall_poly_px[2],
                wall_poly_px[3],
            ]).astype(np.float64)
            tex_img = Image.open(out_png_ortho).convert("RGBA")
            mesh = mesh_by_name.get(name, None)
            if mesh is not None:
                uv = np.empty_like(uvs_px, dtype=np.float64)
                uv[:, 0] = uvs_px[:, 0] / float(out_Wr)
                uv[:, 1] = 1.0 - (uvs_px[:, 1] / float(out_Hr))
                mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)

    stage_timer.record("legacy per-wall fallback loop total", time.perf_counter() - legacy_wall_t0)

    with stage_timer.stage("organize wall artifacts/contact sheets"):
        _save_wall_artifact_folders(
            per_building_out,
            geojson_base,
            viewer_index,
            run_started_at=run_started_at,
            debug_rows=artifact_debug_rows
        )

    if SAVE_VIEWER_INDEX_JSON:
        viewer_bundle_t0 = time.perf_counter()
        # ---- Save viewer index + bundle for LOCAL debugging ----
        index_path = os.path.join(per_building_out, "viewer_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(viewer_index, f, ensure_ascii=False, indent=2)
        print(f"Saved viewer index: {index_path}")

        bundle_path = os.path.join(per_building_out, "viewer_bundle.npz")
        if len(all_wall_quads_global) > 0:
            wall_quads_np = np.stack(all_wall_quads_global, axis=0).astype(np.float64)
        else:
            wall_quads_np = np.zeros((0,4,3), dtype=np.float64)

        save_viewer_bundle_npz(
            bundle_path=bundle_path,
            corners_xyz=corners,
            id_to_idx=id_to_idx,
            edges_by_type=edge_groups,   # this is your edges dict from load_3d_geojson
            wall_quads_xyz_b1b2t2t1=wall_quads_np,
            wall_meta=all_wall_meta_global,
            viewer_index=viewer_index
        )
        print(f"Saved viewer bundle: {bundle_path}")
        stage_timer.record("save viewer index/bundle", time.perf_counter() - viewer_bundle_t0)


    # --------------------- Update roof textures (masked by base loops) ---------------------
    roof_texture_t0 = time.perf_counter()
    if roof_meshes and geotiff_path and Path(geotiff_path).exists():
        try:
            # Closed base loops (in EPSG:25832) for masking
            base_edges_gdf = gdf[gdf['type'] == 'base'].copy()
            roof_loops_all = build_closed_roof_polygons(base_edges_gdf)  # List[Polygon]

            with rasterio.open(geotiff_path) as src:
                width, height = src.width, src.height
                inv = ~src.transform
                rgb = src.read([1, 2, 3])
                rgb = np.moveaxis(rgb, 0, -1).astype(np.uint8)

            # For each roof island, pick the polygon that contains its centroid (fallback: nearest)
            for rname, rmesh, rcoords in roof_meshes:
                # Compute UVs for this roof mesh from GeoTIFF grid
                uv_coords = []
                for x, y in rcoords[:, :2]:
                    col, row = inv * (x, y)
                    u = col / width
                    v = 1.0 - (row / height)
                    uv_coords.append([u, v])
                uv_roof = np.array(uv_coords, dtype=np.float64)

                # Find containing (or nearest) base polygon for masking
                cen = Point(float(rcoords[:, 0].mean()), float(rcoords[:, 1].mean()))
                chosen = None
                best_d = float("inf")
                for poly in roof_loops_all:
                    if poly.contains(cen):
                        chosen = poly
                        break
                    d = poly.distance(cen)
                    if d < best_d:
                        best_d = d
                        chosen = poly

                # Rasterize only the chosen polygon to create alpha for this island
                alpha_mask = rasterize_polygons_to_mask([chosen] if chosen is not None else [],
                                                        width, height, inv)

                # Compose per-island RGBA
                rgba = np.dstack([rgb, alpha_mask]).astype(np.uint8)
                texture_img = Image.fromarray(rgba, mode="RGBA")

                # Assign texture to this roof island
                rmesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv_roof, image=texture_img)

        except Exception as e:
            print(f"WARNING: Roof texture (masked) failed ({e}); keeping white roof.")


    # --------------------- Final base repair + export ---------------------
    stage_timer.record("roof texture update", time.perf_counter() - roof_texture_t0)
    base_repair_t0 = time.perf_counter()
    posttexture_meshes_named = meshes_named
    posttexture_base_repair_info = {
        "version": "1.0",
        "stage": "after_texture_generation_before_final_export",
        "applied": False,
        "reason": "disabled_by_configuration",
    }
    if bool(ENABLE_POSTTEXTURE_BASE_LEVEL_REPAIR):
        finished_wall_records = [
            record
            for records in wall_records_by_loop.values()
            for record in records
        ]
        posttexture_meshes_named, posttexture_base_repair_info = (
            level_finished_building_base(
                meshes_named,
                finished_wall_records,
                level_tolerance_m=POSTTEXTURE_BASE_LEVEL_TOLERANCE_M,
                dominant_color_bits=(
                    POSTTEXTURE_EXTENSION_DOMINANT_COLOR_BITS
                ),
                maximum_color_samples=(
                    POSTTEXTURE_EXTENSION_MAX_COLOR_SAMPLES
                ),
            )
        )
        if posttexture_base_repair_info.get("applied", False):
            print(
                "Post-texture base repair: extended "
                f"{posttexture_base_repair_info['wall_extension_meshes_added']} "
                "wall side(s) to the global minimum z="
                f"{posttexture_base_repair_info['minimum_base_z_m']:.3f} m "
                "and replaced/created "
                f"{posttexture_base_repair_info['base_meshes_replaced'] + posttexture_base_repair_info['base_meshes_created']} "
                "flat base surface(s)."
            )
            if not posttexture_base_repair_info.get(
                "geometry_complete", False
            ):
                print(
                    "WARNING: Post-texture base repair could not close every "
                    "required wall/base fragment; inspect "
                    "posttexture_base_repair.json."
                )
        else:
            print(
                "Post-texture base repair made no geometry changes "
                f"({posttexture_base_repair_info.get('reason', 'unknown')})."
            )
    base_repair_path = Path(per_building_out) / "posttexture_base_repair.json"
    with open(base_repair_path, "w", encoding="utf-8") as f:
        json.dump(
            posttexture_base_repair_info,
            f,
            ensure_ascii=False,
            indent=2,
        )
    stage_timer.record(
        "post-texture base level repair",
        time.perf_counter() - base_repair_t0,
    )

    export_t0 = time.perf_counter()
    export_source_meshes_named, topology_repair_info = repair_mesh_t_junctions(
        posttexture_meshes_named,
    )
    if (
        topology_repair_info["degenerate_faces_removed"]
        or topology_repair_info["boundary_edges_split"]
    ):
        print(
            "Repaired export mesh topology: "
            f"removed {topology_repair_info['degenerate_faces_removed']} "
            "zero-area face(s), split "
            f"{topology_repair_info['boundary_edges_split']} "
            "T-junction edge(s)."
        )

    if export_source_meshes_named:
        geojson_stem = os.path.splitext(os.path.basename(geojson_path))[0]
        export_origin_epsg = None
        export_meshes_named = export_source_meshes_named
        if GLB_EXPORT_LOCAL_COORDINATES:
            export_origin_epsg = _make_export_origin(
                export_source_meshes_named,
                relative_to_ground=False,
            )
            if export_origin_epsg is not None:
                export_meshes_named = _copy_meshes_for_local_y_up(
                    export_source_meshes_named,
                    export_origin_epsg,
                )

        scene = trimesh.Scene()
        for name, m in export_meshes_named:
            scene.add_geometry(m, node_name=name)
        glb_path = Path(per_building_out) / name_for("glb", base=geojson_stem)
        scene.export(glb_path)
        asset_extras = {
            "crs": SOURCE_CRS,
            "mesh_topology_repair": topology_repair_info,
            "posttexture_base_repair": posttexture_base_repair_info,
        }
        if GLB_EXPORT_LOCAL_COORDINATES and export_origin_epsg is not None:
            asset_extras.update({
                "coordinates": "local_gltf_y_up",
                "axis_mapping": "gltf_x=east,gltf_y=up,gltf_z=-north",
                "coordinate_origin_epsg_25832": [float(v) for v in export_origin_epsg],
            })
        else:
            asset_extras["coordinates"] = f"direct_{str(SOURCE_CRS).lower()}"
        if patch_glb_materials_double_sided(glb_path, asset_extras=asset_extras):
            print("Patched GLB materials to opaque, matte, and double-sided.")
        print(f"\nExported textured GLB: {glb_path}")
        if EXPORT_KMZ:
            kmz_path = Path(per_building_out) / name_for("kmz", base=geojson_stem)
            saved_kmz = _save_textured_kmz(
                export_source_meshes_named,
                kmz_path,
                name=geojson_stem,
            )
            if saved_kmz is not None:
                print(f"Exported textured KMZ: {saved_kmz}")
    else:
        print("WARNING: No geometry to export (no meshes built).")

    stage_timer.record("export GLB/KMZ", time.perf_counter() - export_t0)
    stage_timer.finish(per_building_out)

def main():
    ensure_outdir(OUTPUT_DIR)
    geojson_dir = Path(GEOJSON_DIR)
    geotiff_dir = Path(GEOTIFF_DIR)

    files = sorted(geojson_dir.glob("*.geojson"))
    if not files:
        print(f"No .geojson files found in: {geojson_dir}")
        return

    print(f"Found {len(files)} .geojson files. Starting batch...")

    sam_load_t0 = time.perf_counter()
    device, processor, sam3_prompt_facade, sam3_prompt_roof = load_sam3(
    prompt_facade=SAM3_PROMPT_FACADE,
    prompt_roof=SAM3_PROMPT_ROOF
    )
    sam3_prompt_facade_refinement = globals().get("SAM3_PROMPT_FACADE_REFINEMENT", None)
    print(f"[time] batch | SAM3 model load: {time.perf_counter() - sam_load_t0:.2f}s")
    print(
        f"SAM3 loaded once on device: {device} | "
        f"facade_prompt={sam3_prompt_facade!r} | "
        f"facade_refinement_prompt={sam3_prompt_facade_refinement!r} | "
        f"roof_prompt={sam3_prompt_roof!r}"
    )



    for idx, gj in enumerate(files, 1):
        base = gj.stem  # e.g., "building_267160681_3d"
        roof_base = base[:-3] if base.endswith("_3d") else base
        tif_path  = geotiff_dir / f"{roof_base}.tif"
        tif_path_alt = geotiff_dir / f"{roof_base}.tiff"

        geotiff_for_this = None
        if tif_path.exists():
            geotiff_for_this = str(tif_path)
        elif tif_path_alt.exists():
            geotiff_for_this = str(tif_path_alt)
        else:
            print(f"WARNING: No matching GeoTIFF for {base} in {geotiff_dir} (looked for {roof_base}.tif/.tiff). Roof will remain white.")

        print(f"\n[{idx}/{len(files)}] Processing: {gj.name}")
        try:
            process_building(
                str(gj),
                OUTPUT_DIR,
                geotiff_path=geotiff_for_this,
                device=device,
                processor=processor,
                sam3_prompt_facade=sam3_prompt_facade,
                sam3_prompt_facade_refinement=sam3_prompt_facade_refinement,
                sam3_prompt_roof=sam3_prompt_roof
            )

            # Optional: free cached memory between buildings (useful on GPU)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"ERROR: Failed on {gj.name}: {e}")
            traceback.print_exc()


    print("\nBatch complete.")
