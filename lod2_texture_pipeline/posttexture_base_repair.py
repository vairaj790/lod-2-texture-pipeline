# -*- coding: utf-8 -*-
"""Final-export repair for uneven LoD-2 building bases.

The texture pipeline must use the source wall geometry unchanged while it
selects imagery and rectifies facade textures.  This module therefore operates
only on the finished meshes: it leaves every original wall in place, adds a
solid-colour strip below walls whose base is above the building-wide minimum,
and recreates the underside at that minimum elevation.
"""

from collections import defaultdict

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import triangulate


DEFAULT_EXTENSION_RGBA = (240, 240, 240, 255)
DEFAULT_BASE_RGBA = (240, 240, 240, 255)


def _texture_image(mesh):
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
        return image if image.mode == "RGBA" else image.convert("RGBA")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        return None
    return Image.fromarray(array.astype(np.uint8)).convert("RGBA")


def _valid_uv(mesh):
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
        return None
    return uv


def _mode_rgba(colors, fallback_rgba=DEFAULT_EXTENSION_RGBA):
    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    if colors.ndim != 2 or colors.shape[0] == 0 or colors.shape[1] < 3:
        return tuple(int(v) for v in fallback_rgba)
    rgb = np.clip(colors[:, :3], 0, 255).astype(np.uint8)
    unique, counts = np.unique(rgb, axis=0, return_counts=True)
    winner = unique[int(np.argmax(counts))]
    return int(winner[0]), int(winner[1]), int(winner[2]), 255


def _visual_fallback_rgba(mesh, fallback_rgba=DEFAULT_EXTENSION_RGBA):
    visual = getattr(mesh, "visual", None)
    kind = str(getattr(visual, "kind", "") or "").lower()
    if kind == "face":
        colors = getattr(visual, "face_colors", None)
        if colors is not None:
            return _mode_rgba(colors, fallback_rgba)
    if kind == "vertex":
        colors = getattr(visual, "vertex_colors", None)
        if colors is not None:
            return _mode_rgba(colors, fallback_rgba)

    material = getattr(visual, "material", None)
    for attr in ("main_color", "diffuse", "baseColorFactor"):
        color = getattr(material, attr, None) if material is not None else None
        if color is None:
            continue
        color = np.asarray(color).reshape(-1)
        if len(color) < 3 or not np.isfinite(color[:3]).all():
            continue
        rgb = color[:3].astype(np.float64)
        if float(np.max(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return int(rgb[0]), int(rgb[1]), int(rgb[2]), 255
    return tuple(int(v) for v in fallback_rgba)


def dominant_wall_texture_rgba(
    mesh,
    *,
    quantization_bits=5,
    maximum_samples=500_000,
    minimum_alpha=1,
    fallback_rgba=DEFAULT_EXTENSION_RGBA,
):
    """Return a representative modal colour from pixels covered by ``mesh`` UVs.

    Photographic pixels rarely repeat exactly.  RGB is consequently grouped
    into a small deterministic histogram (five bits per channel by default),
    and the median original RGB in the most populated group is returned.  Only
    opaque pixels inside the mesh's UV triangles participate, so transparent
    texture gutters and other walls in a shared facade atlas cannot win.
    """
    fallback = _visual_fallback_rgba(mesh, fallback_rgba)
    image = _texture_image(mesh)
    uv = _valid_uv(mesh)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
    if (
        image is None
        or uv is None
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(faces) == 0
        or np.any(faces < 0)
        or np.any(faces >= len(uv))
    ):
        return fallback, {
            "source": "visual_fallback",
            "covered_opaque_pixels": 0,
            "sampled_pixels": 0,
        }

    width, height = image.size
    if width <= 0 or height <= 0:
        return fallback, {
            "source": "visual_fallback",
            "covered_opaque_pixels": 0,
            "sampled_pixels": 0,
        }

    uv_pixels = np.empty_like(uv, dtype=np.float64)
    uv_pixels[:, 0] = uv[:, 0] * float(width)
    uv_pixels[:, 1] = (1.0 - uv[:, 1]) * float(height)
    face_pixels = uv_pixels[faces]
    finite_faces = np.isfinite(face_pixels).all(axis=(1, 2))
    face_pixels = face_pixels[finite_faces]
    if len(face_pixels) == 0:
        return fallback, {
            "source": "visual_fallback",
            "covered_opaque_pixels": 0,
            "sampled_pixels": 0,
        }

    min_x = max(0, int(np.floor(np.min(face_pixels[:, :, 0]))))
    min_y = max(0, int(np.floor(np.min(face_pixels[:, :, 1]))))
    max_x = min(width - 1, int(np.ceil(np.max(face_pixels[:, :, 0]))))
    max_y = min(height - 1, int(np.ceil(np.max(face_pixels[:, :, 1]))))
    if min_x > max_x or min_y > max_y:
        return fallback, {
            "source": "visual_fallback",
            "covered_opaque_pixels": 0,
            "sampled_pixels": 0,
        }

    crop_width = max_x - min_x + 1
    crop_height = max_y - min_y + 1
    mask_image = Image.new("L", (crop_width, crop_height), 0)
    draw = ImageDraw.Draw(mask_image, "L")
    offset = np.array([float(min_x), float(min_y)], dtype=np.float64)
    for triangle in face_pixels:
        points = [tuple(float(v) for v in point) for point in triangle - offset]
        draw.polygon(points, fill=255)

    rgba = np.asarray(
        image.crop((min_x, min_y, max_x + 1, max_y + 1)),
        dtype=np.uint8,
    )
    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    valid = mask & (rgba[:, :, 3] >= int(minimum_alpha))
    pixels = rgba[valid]
    opaque_count = int(len(pixels))
    sampling_source = "uv_triangle_texture_mode"
    if len(pixels) == 0:
        # A fully transparent final texture can still carry the rectified RGB
        # below alpha.  Restrict this fallback to non-black pixels inside the
        # wall UVs; transparent gutters outside the UV triangles remain out.
        covered = rgba[mask]
        pixels = covered[np.any(covered[:, :3] != 0, axis=1)]
        sampling_source = "uv_triangle_hidden_rgb_mode"
    candidate_count = int(len(pixels))
    if candidate_count == 0:
        return fallback, {
            "source": "visual_fallback",
            "covered_opaque_pixels": 0,
            "sampled_pixels": 0,
        }

    maximum_samples = max(1, int(maximum_samples))
    if len(pixels) > maximum_samples:
        sample_indices = np.linspace(
            0,
            len(pixels) - 1,
            num=maximum_samples,
            dtype=np.int64,
        )
        sampled = pixels[sample_indices]
    else:
        sampled = pixels

    bits = min(8, max(1, int(quantization_bits)))
    shift = 8 - bits
    rgb = sampled[:, :3].astype(np.uint32)
    quantized = rgb >> shift
    keys = (
        (quantized[:, 0] << (2 * bits))
        | (quantized[:, 1] << bits)
        | quantized[:, 2]
    )
    unique_keys, counts = np.unique(keys, return_counts=True)
    winning_key = unique_keys[int(np.argmax(counts))]
    winning_pixels = sampled[keys == winning_key, :3]
    color = np.clip(
        np.rint(np.median(winning_pixels.astype(np.float64), axis=0)),
        0,
        255,
    ).astype(np.uint8)
    rgba_out = int(color[0]), int(color[1]), int(color[2]), 255
    return rgba_out, {
        "source": sampling_source,
        "covered_opaque_pixels": opaque_count,
        "candidate_pixels": candidate_count,
        "sampled_pixels": int(len(sampled)),
        "quantization_bits": bits,
        "winning_bin_pixels": int(len(winning_pixels)),
    }


def _is_base_mesh_name(name):
    lowered = str(name).strip().lower()
    return lowered == "base" or lowered.startswith("base_")


def _finite_wall_records(wall_records):
    valid = []
    for record in wall_records or []:
        quad = np.asarray(record.get("wall_quad", []), dtype=np.float64)
        if quad.shape != (4, 3) or not np.isfinite(quad).all():
            continue
        valid.append((record, quad))
    return valid


def _base_elevations(meshes_named, valid_records):
    values = [quad[:2, 2] for _record, quad in valid_records]
    for name, mesh in meshes_named:
        if not _is_base_mesh_name(name):
            continue
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            continue
        finite = np.isfinite(vertices[:, 2])
        if finite.any():
            values.append(vertices[finite, 2])
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])


def _wall_outward_normal(record, source_mesh):
    normal = np.asarray(record.get("normal", []), dtype=np.float64).reshape(-1)
    if len(normal) >= 3 and np.isfinite(normal[:3]).all():
        normal = normal[:3]
    else:
        normal = np.zeros(3, dtype=np.float64)
    if float(np.linalg.norm(normal)) <= 1.0e-12:
        try:
            normals = np.asarray(source_mesh.face_normals, dtype=np.float64)
            normal = np.nanmean(normals, axis=0)
        except Exception:
            normal = np.zeros(3, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    return normal / length if length > 1.0e-12 else normal


def _make_wall_extension_mesh(
    record,
    quad,
    source_mesh,
    minimum_z,
    rgba,
    *,
    area_tolerance=1.0e-12,
):
    top_1 = quad[0].copy()
    top_2 = quad[1].copy()
    low_1 = top_1.copy()
    low_2 = top_2.copy()
    low_1[2] = float(minimum_z)
    low_2[2] = float(minimum_z)
    vertices = np.vstack([low_1, low_2, top_2, top_1]).astype(np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    triangles = vertices[faces]
    twice_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    faces = faces[np.isfinite(twice_area) & (twice_area > float(area_tolerance))]
    if len(faces) == 0:
        return None

    expected = _wall_outward_normal(record, source_mesh)
    combined_normal = np.sum(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=0,
    )
    if float(np.dot(combined_normal, expected)) < 0.0:
        faces = faces[:, [0, 2, 1]]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.visual.face_colors = np.tile(
        np.asarray(rgba, dtype=np.uint8).reshape(1, 4),
        (len(faces), 1),
    )
    mesh.metadata.update({
        "posttexture_base_extension": True,
        "source_wall_mesh": str(record.get("mesh_name")),
        "minimum_base_z_m": float(minimum_z),
        "dominant_wall_rgba": [int(value) for value in rgba],
    })
    return mesh


def _flatten_base_mesh(mesh, minimum_z):
    output = mesh.copy()
    vertices = np.asarray(output.vertices, dtype=np.float64).copy()
    vertices[:, 2] = float(minimum_z)
    output.vertices = vertices
    faces = np.asarray(output.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    keep = np.isfinite(cross).all(axis=1) & (
        np.linalg.norm(cross, axis=1) > 1.0e-12
    )
    if not keep.any():
        return None
    if not keep.all():
        output.update_faces(keep)
        faces = np.asarray(output.faces, dtype=np.int64).copy()
        triangles = vertices[faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
    else:
        faces = faces.copy()
    upward = cross[:, 2] > 0.0
    if upward.any():
        faces[upward] = faces[upward][:, [0, 2, 1]]
        output.faces = faces
    output.metadata.update({
        "posttexture_flat_base": True,
        "minimum_base_z_m": float(minimum_z),
    })
    return output


def _new_flat_bases_from_wall_records(valid_records, minimum_z):
    grouped = defaultdict(list)
    for record, quad in valid_records:
        key = (record.get("component_id"), record.get("loop_id"))
        grouped[key].append((record, quad))

    meshes = []
    for group_index, (key, rows) in enumerate(sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    )):
        rows.sort(key=lambda item: int(item[0].get("loop_index", 0)))
        xy = np.asarray([quad[0, :2] for _record, quad in rows], dtype=np.float64)
        if len(xy) < 3 or not np.isfinite(xy).all():
            continue
        polygon = Polygon(xy)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygons = (
            list(polygon.geoms)
            if getattr(polygon, "geom_type", "") == "MultiPolygon"
            else [polygon]
        )
        for part_index, part in enumerate(polygons):
            if part.is_empty or float(part.area) <= 1.0e-12:
                continue
            vertices = []
            faces = []
            for triangle in triangulate(part):
                if not part.covers(triangle):
                    continue
                triangle_xy = np.asarray(triangle.exterior.coords[:-1], dtype=np.float64)
                if triangle_xy.shape != (3, 2):
                    continue
                first = len(vertices)
                vertices.extend([
                    [float(x), float(y), float(minimum_z)]
                    for x, y in triangle_xy
                ])
                # Shapely triangles are counter-clockwise; the underside faces
                # down so its winding closes the shell opposite the roof.
                faces.append([first, first + 2, first + 1])
            if not faces:
                continue
            mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            mesh.visual.face_colors = np.tile(
                np.asarray(DEFAULT_BASE_RGBA, dtype=np.uint8).reshape(1, 4),
                (len(mesh.faces), 1),
            )
            mesh.metadata.update({
                "posttexture_flat_base": True,
                "minimum_base_z_m": float(minimum_z),
                "reconstructed_from_wall_loop": True,
            })
            component_id, loop_id = key
            name = (
                "base_posttexture_flat_"
                f"c{component_id}_l{loop_id}_g{group_index:02d}_p{part_index:02d}"
            )
            meshes.append((name, mesh))
    return meshes


def level_finished_building_base(
    meshes_named,
    wall_records,
    *,
    level_tolerance_m=0.001,
    dominant_color_bits=5,
    maximum_color_samples=500_000,
    fallback_rgba=DEFAULT_EXTENSION_RGBA,
):
    """Level an uneven base after textures are final and before mesh export.

    Original wall meshes are never moved or re-UVed.  If all base elevations
    already agree within ``level_tolerance_m``, the original list object is
    returned unchanged.  Otherwise newly coloured wall-extension meshes and a
    flat replacement underside are returned with an audit dictionary.
    """
    meshes_named = list(meshes_named) if not isinstance(meshes_named, list) else meshes_named
    wall_records = list(wall_records or [])
    valid_records = _finite_wall_records(wall_records)
    elevations = _base_elevations(meshes_named, valid_records)
    tolerance = max(0.0, float(level_tolerance_m))
    info = {
        "version": "1.0",
        "stage": "after_texture_generation_before_final_export",
        "applied": False,
        "geometry_complete": False,
        "reason": "not_evaluated",
        "level_tolerance_m": tolerance,
        "base_point_count": int(len(elevations)),
        "valid_wall_records": int(len(valid_records)),
        "malformed_wall_records": int(
            len(wall_records or []) - len(valid_records)
        ),
        "minimum_base_z_m": None,
        "maximum_base_z_m": None,
        "base_relief_m": None,
        "base_meshes_replaced": 0,
        "base_meshes_removed": 0,
        "base_meshes_created": 0,
        "wall_extension_meshes_added": 0,
        "wall_extension_triangles_added": 0,
        "maximum_extension_height_m": 0.0,
        "wall_records_missing_export_mesh": [],
        "wall_extensions": [],
    }
    if len(elevations) == 0:
        info["reason"] = "no_finite_base_points"
        return meshes_named, info
    if not valid_records:
        info["reason"] = "no_valid_wall_records"
        return meshes_named, info

    minimum_z = float(np.min(elevations))
    maximum_z = float(np.max(elevations))
    relief = float(maximum_z - minimum_z)
    info.update({
        "minimum_base_z_m": minimum_z,
        "maximum_base_z_m": maximum_z,
        "base_relief_m": relief,
    })
    if relief <= tolerance:
        info["reason"] = "already_level"
        info["geometry_complete"] = True
        return meshes_named, info

    mesh_lookup = {str(name): mesh for name, mesh in meshes_named}
    walls_requiring_extension = int(sum(
        float(np.max(quad[:2, 2] - minimum_z)) > 1.0e-12
        for _record, quad in valid_records
    ))
    info["wall_records_requiring_extension"] = walls_requiring_extension
    extension_rows = []
    extension_meshes = []
    used_names = {str(name) for name, _mesh in meshes_named}
    for record, quad in valid_records:
        raw_mesh_name = record.get("mesh_name")
        if not raw_mesh_name:
            info["wall_records_missing_export_mesh"].append(
                f"wall_record_{record.get('global_index', 'unknown')}"
            )
            continue
        mesh_name = str(raw_mesh_name)
        source_mesh = mesh_lookup.get(mesh_name)
        if source_mesh is None:
            info["wall_records_missing_export_mesh"].append(mesh_name)
            continue
        if float(np.max(quad[:2, 2] - minimum_z)) <= 1.0e-12:
            continue
        rgba, color_info = dominant_wall_texture_rgba(
            source_mesh,
            quantization_bits=dominant_color_bits,
            maximum_samples=maximum_color_samples,
            fallback_rgba=fallback_rgba,
        )
        extension = _make_wall_extension_mesh(
            record,
            quad,
            source_mesh,
            minimum_z,
            rgba,
        )
        if extension is None:
            continue
        extension_name = f"{mesh_name}__terrain_skirt"
        suffix = 1
        while extension_name in used_names:
            extension_name = f"{mesh_name}__terrain_skirt_{suffix:02d}"
            suffix += 1
        used_names.add(extension_name)
        extension_meshes.append((extension_name, extension))
        extension_rows.append({
            "mesh_name": extension_name,
            "source_wall_mesh": mesh_name,
            "source_bottom_z_m": [float(value) for value in quad[:2, 2]],
            "extension_height_m": [
                float(max(0.0, value - minimum_z)) for value in quad[:2, 2]
            ],
            "rgba": [int(value) for value in rgba],
            "color_sampling": color_info,
            "triangles": int(len(extension.faces)),
        })

    base_entries = [
        (name, mesh) for name, mesh in meshes_named if _is_base_mesh_name(name)
    ]
    flattened_bases = [
        (name, _flatten_base_mesh(mesh, minimum_z))
        for name, mesh in base_entries
    ]
    reuse_flattened_bases = bool(base_entries) and all(
        mesh is not None for _name, mesh in flattened_bases
    )
    flat_base_lookup = dict(flattened_bases) if reuse_flattened_bases else {}

    output = []
    for name, mesh in meshes_named:
        if not _is_base_mesh_name(name):
            output.append((name, mesh))
        elif reuse_flattened_bases:
            output.append((name, flat_base_lookup[name]))

    created_bases = []
    if not reuse_flattened_bases:
        created_bases = _new_flat_bases_from_wall_records(valid_records, minimum_z)
        output.extend(created_bases)
    output.extend(extension_meshes)

    flat_base_count = (
        len(base_entries) if reuse_flattened_bases else len(created_bases)
    )
    geometry_complete = bool(flat_base_count) and (
        len(extension_meshes) == walls_requiring_extension
    )
    info.update({
        "applied": True,
        "geometry_complete": bool(geometry_complete),
        "reason": (
            "uneven_base_extended_to_global_minimum"
            if geometry_complete
            else "uneven_base_repair_incomplete"
        ),
        "base_meshes_replaced": int(len(base_entries) if reuse_flattened_bases else 0),
        "base_meshes_removed": int(len(base_entries)),
        "base_meshes_created": int(len(created_bases)),
        "wall_extension_meshes_added": int(len(extension_meshes)),
        "wall_extension_triangles_added": int(sum(
            len(mesh.faces) for _name, mesh in extension_meshes
        )),
        "maximum_extension_height_m": float(max(
            (
                max(row["extension_height_m"])
                for row in extension_rows
            ),
            default=0.0,
        )),
        "wall_extensions": extension_rows,
    })
    return output, info
