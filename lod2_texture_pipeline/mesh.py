# -*- coding: utf-8 -*-
"""Mesh building, triangulation, roof polygonization, and raster masking."""

from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize_full, triangulate


def repair_mesh_t_junctions(
    meshes_named,
    tolerance=1e-6,
    area_tolerance=1e-12,
):
    """Return export-safe meshes with conforming shared boundary edges.

    LOD2 inputs can represent a roof-height discontinuity with a short wall
    edge ``B-L``, a roof-seam edge ``L-H``, and the neighbouring wall's single
    tall edge ``B-H``.  The surfaces occupy the right locations, but the latter
    is a T-junction and triangle-soup watertightness checks see three open
    edges.  Split such boundary edges at already-existing vertices from the
    other meshes.  No surface is moved; texture coordinates and vertex colors
    are linearly interpolated along the split edge.

    Zero-area faces are removed first.  A mesh containing only degenerate
    faces is omitted from the returned list.
    """
    tol = max(float(tolerance), 1e-12)
    area_tol = max(float(area_tolerance), 0.0)
    cleaned = []
    stats = {
        "input_meshes": int(len(meshes_named)),
        "output_meshes": 0,
        "degenerate_faces_removed": 0,
        "degenerate_meshes_removed": 0,
        "boundary_edges_split": 0,
        "vertices_added": 0,
        "faces_added": 0,
    }

    for name, mesh in meshes_named:
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
        faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[1] != 3
            or len(faces) == 0
        ):
            stats["degenerate_meshes_removed"] += 1
            continue

        triangles = vertices[faces]
        twice_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        keep = np.isfinite(twice_area) & (twice_area > area_tol)
        removed = int((~keep).sum())
        stats["degenerate_faces_removed"] += removed
        if not keep.any():
            stats["degenerate_meshes_removed"] += 1
            continue

        cleaned.append({
            "name": name,
            "mesh": mesh,
            "vertices": vertices.copy(),
            "faces": faces[keep].copy(),
            "source_face_indices": np.flatnonzero(keep).astype(np.int64),
            "removed_faces": removed,
        })

    if not cleaned:
        return [], stats

    candidate_positions = []
    candidate_mesh_indices = []
    for mesh_index, item in enumerate(cleaned):
        candidate_positions.append(item["vertices"])
        candidate_mesh_indices.extend([mesh_index] * len(item["vertices"]))
    candidate_positions = np.vstack(candidate_positions)
    candidate_mesh_indices = np.asarray(candidate_mesh_indices, dtype=np.int64)

    repaired = []
    for mesh_index, item in enumerate(cleaned):
        old_mesh = item["mesh"]
        vertices = item["vertices"]
        faces = item["faces"]

        all_edges = np.sort(
            faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(
            all_edges,
            axis=0,
            return_counts=True,
        )
        boundary_edges = unique_edges[edge_counts == 1]

        split_positions = {}
        other_candidates = candidate_positions[
            candidate_mesh_indices != mesh_index
        ]
        for edge in boundary_edges:
            a_idx, b_idx = int(edge[0]), int(edge[1])
            a = vertices[a_idx]
            b = vertices[b_idx]
            direction = b - a
            length_sq = float(np.dot(direction, direction))
            if length_sq <= tol * tol:
                continue

            # Elementwise dot products avoid launching a BLAS kernel for the
            # many tiny (N x 3) boundary-candidate checks.
            delta = other_candidates - a
            t = np.sum(delta * direction, axis=1) / length_sq
            interior = (t > tol / np.sqrt(length_sq)) & (
                t < 1.0 - tol / np.sqrt(length_sq)
            )
            if not interior.any():
                continue
            projected = a + t[:, None] * direction
            distance = np.linalg.norm(projected - other_candidates, axis=1)
            matches = np.flatnonzero(interior & (distance <= tol))
            if len(matches) == 0:
                continue

            ordered = sorted(
                [
                    (float(t[index]), other_candidates[index].copy())
                    for index in matches
                ],
                key=lambda pair: pair[0],
            )
            deduped = []
            for fraction, position in ordered:
                if deduped and np.linalg.norm(position - deduped[-1][1]) <= tol:
                    continue
                deduped.append((fraction, position))
            if deduped:
                split_positions[(a_idx, b_idx)] = deduped

        if not split_positions and item["removed_faces"] == 0:
            # The caller only needs copies for meshes whose connectivity
            # changes. Reusing untouched meshes also avoids duplicating large
            # baked texture images in memory before export.
            repaired.append((item["name"], old_mesh))
            continue

        new_vertices = vertices.tolist()
        visual = getattr(old_mesh, "visual", None)
        old_uv = getattr(visual, "uv", None)
        if old_uv is not None:
            old_uv = np.asarray(old_uv, dtype=np.float64)
            if old_uv.shape != (len(vertices), 2):
                old_uv = None
        new_uv = old_uv.tolist() if old_uv is not None else None

        visual_kind = getattr(visual, "kind", None)
        old_vertex_colors = None
        new_vertex_colors = None
        if visual_kind == "vertex":
            old_vertex_colors = np.asarray(visual.vertex_colors)
            if len(old_vertex_colors) == len(vertices):
                new_vertex_colors = old_vertex_colors.astype(np.float64).tolist()
            else:
                old_vertex_colors = None

        edge_middle_indices = {}
        for edge, points in sorted(split_positions.items()):
            a_idx, b_idx = edge
            middle = []
            for fraction, position in points:
                new_index = len(new_vertices)
                new_vertices.append(position.tolist())
                if new_uv is not None:
                    interpolated_uv = (
                        (1.0 - fraction) * old_uv[a_idx]
                        + fraction * old_uv[b_idx]
                    )
                    new_uv.append(interpolated_uv.tolist())
                if new_vertex_colors is not None:
                    color = (
                        (1.0 - fraction) * old_vertex_colors[a_idx]
                        + fraction * old_vertex_colors[b_idx]
                    )
                    new_vertex_colors.append(color.tolist())
                middle.append(new_index)
            edge_middle_indices[edge] = middle
            stats["boundary_edges_split"] += 1
            stats["vertices_added"] += len(middle)

        face_records = [
            ([int(v) for v in face], int(source_face_index))
            for face, source_face_index in zip(
                faces,
                item["source_face_indices"],
            )
        ]
        for edge, middle in sorted(edge_middle_indices.items()):
            a_idx, b_idx = edge
            updated = []
            for face, source_face_index in face_records:
                split = False
                for offset in range(3):
                    first = face[offset]
                    second = face[(offset + 1) % 3]
                    opposite = face[(offset + 2) % 3]
                    if first == a_idx and second == b_idx:
                        chain = [a_idx, *middle, b_idx]
                    elif first == b_idx and second == a_idx:
                        chain = [b_idx, *reversed(middle), a_idx]
                    else:
                        continue
                    updated.extend(
                        ([chain[i], chain[i + 1], opposite], source_face_index)
                        for i in range(len(chain) - 1)
                    )
                    split = True
                    break
                if not split:
                    updated.append((face, source_face_index))
            face_records = updated

        new_faces = np.asarray([record[0] for record in face_records], dtype=np.int64)
        source_face_indices = np.asarray(
            [record[1] for record in face_records],
            dtype=np.int64,
        )
        stats["faces_added"] += int(len(new_faces) - len(faces))

        out = trimesh.Trimesh(
            vertices=np.asarray(new_vertices, dtype=np.float64),
            faces=new_faces,
            process=False,
        )
        if new_uv is not None:
            out.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.asarray(new_uv, dtype=np.float64),
                material=getattr(visual, "material", None),
            )
        elif new_vertex_colors is not None:
            out.visual.vertex_colors = np.clip(
                np.rint(np.asarray(new_vertex_colors)),
                0,
                255,
            ).astype(np.uint8)
        elif visual_kind == "face":
            old_face_colors = np.asarray(visual.face_colors)
            if len(old_face_colors) == len(old_mesh.faces):
                out.visual.face_colors = old_face_colors[source_face_indices]

        out.metadata.update(dict(getattr(old_mesh, "metadata", {}) or {}))
        repaired.append((item["name"], out))

    stats["output_meshes"] = int(len(repaired))
    return repaired, stats


def _build_wall_mesh_from_verts(verts4_xyz: np.ndarray,
                                outward_normal_xyz: np.ndarray,
                                uv_px: Optional[np.ndarray] = None,
                                tex_img: Optional["Image.Image"] = None,
                                out_w: Optional[int] = None,
                                out_h: Optional[int] = None,
                                flat_rgba=(240, 240, 240, 255)) -> "trimesh.Trimesh":
    import numpy as np, trimesh
    v = np.asarray(verts4_xyz, dtype=np.float64)  # [b1,b2,t2,t1]
    face_normal = np.cross(v[1] - v[0], v[2] - v[0])
    if np.dot(face_normal, np.asarray(outward_normal_xyz, dtype=np.float64)) < 0:
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    else:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    if tex_img is not None and uv_px is not None and out_w and out_h:
        uv = np.empty_like(uv_px, dtype=np.float64)
        uv[:, 0] = uv_px[:, 0] / float(out_w)
        uv[:, 1] = 1.0 - (uv_px[:, 1] / float(out_h))  # flip V for glTF
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)
    else:
        r, g, b, a = flat_rgba
        mesh.visual.face_colors = [r, g, b, a]
    return mesh

def triangulate_surface(edges, corners, id_to_idx, split_components=False):
    import numpy as np
    from scipy.spatial import Delaunay
    from collections import defaultdict, deque

    # Build adjacency to discover connected components from the provided edges
    used_ids = set()
    adj = defaultdict(set)
    for s, t in edges:
        used_ids.update([s, t])
        adj[s].add(t)
        adj[t].add(s)
    used_ids = sorted(used_ids)
    if not used_ids:
        return (None, None) if not split_components else ([], [])

    # Fast path (previous behavior): one global triangulation
    if not split_components:
        coords = np.array([[*corners[id_to_idx[i]]] for i in used_ids], dtype=np.float64)
        if len(coords) < 3:
            return None, None
        tri = Delaunay(coords[:, :2])
        faces = tri.simplices
        return coords, faces

    # Split into connected components (so separate islands don't get bridged)
    comps = []
    seen = set()
    for v in used_ids:
        if v in seen:
            continue
        q = deque([v])
        seen.add(v)
        comp = [v]
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
                    comp.append(w)
        comps.append(sorted(comp))

    coords_list, faces_list = [], []
    for comp in comps:
        coords = np.array([[*corners[id_to_idx[i]]] for i in comp], dtype=np.float64)
        if len(coords) < 3:
            continue
        tri = Delaunay(coords[:, :2])
        faces = tri.simplices
        coords_list.append(coords)
        faces_list.append(faces)

    return coords_list, faces_list

def dedupe_ring_vertex_indices(vertex_indices):
    ring = [int(v) for v in vertex_indices]
    if len(ring) == 0:
        return []

    deduped = [ring[0]]
    for vi in ring[1:]:
        if vi != deduped[-1]:
            deduped.append(vi)

    if len(deduped) >= 2 and deduped[0] == deduped[-1]:
        deduped = deduped[:-1]

    return deduped

def triangulate_planar_ring_3d(coords3d):
    arr = np.asarray(coords3d, dtype=float)
    if arr.shape[0] < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]

    if arr.shape[0] < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    centroid = arr.mean(axis=0)
    centered = arr - centroid

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    axis_u = vh[0]
    axis_v = vh[1]
    if np.linalg.norm(axis_u) < 1e-12 or np.linalg.norm(axis_v) < 1e-12:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)

    uv = np.column_stack([centered @ axis_u, centered @ axis_v])
    poly_2d = Polygon(uv)
    if not poly_2d.is_valid:
        poly_2d = poly_2d.buffer(0)

    if poly_2d.is_empty:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    if poly_2d.geom_type == "MultiPolygon":
        poly_2d = max(poly_2d.geoms, key=lambda g: g.area)

    if poly_2d.area <= 1e-12:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    scale = max(1.0, float(np.max(np.ptp(uv, axis=0))))
    lookup_tol = 1e-7 * scale

    vertices = []
    triangles_idx = []

    for tri in triangulate(poly_2d):
        if not poly_2d.buffer(1e-9 * scale).covers(tri):
            continue

        tri_uv = np.asarray(tri.exterior.coords[:-1], dtype=float)
        if tri_uv.shape != (3, 2):
            continue

        base_idx = len(vertices)
        for x, y in tri_uv:
            d2 = np.sum((uv - np.array([x, y], dtype=float)) ** 2, axis=1)
            nearest_idx = int(np.argmin(d2))
            if float(np.sqrt(d2[nearest_idx])) <= lookup_tol:
                xyz = arr[nearest_idx]
            else:
                xyz = centroid + (x * axis_u) + (y * axis_v)
            vertices.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

        triangles_idx.append([base_idx, base_idx + 1, base_idx + 2])

    if len(triangles_idx) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    return np.asarray(vertices, dtype=float), np.asarray(triangles_idx, dtype=np.int32)

def build_trimesh_from_surface_face(corners, surface_face, flat_rgba=(220, 220, 220, 255)):
    ring = dedupe_ring_vertex_indices(surface_face.get("vertex_indices", []))
    if len(ring) < 3:
        return None, None

    corners_arr = np.asarray(corners, dtype=np.float64)
    if any(v < 0 or v >= len(corners_arr) for v in ring):
        return None, None

    coords3d = corners_arr[ring]
    if not np.all(np.isfinite(coords3d)):
        return None, None

    verts_i, tris_i = triangulate_planar_ring_3d(coords3d)
    if tris_i.shape[0] == 0:
        return None, None

    mesh = trimesh.Trimesh(vertices=verts_i, faces=tris_i, process=False)
    mesh.visual.face_colors = flat_rgba
    return mesh, verts_i

def build_closed_roof_polygons(base_edges_gdf) -> List[Polygon]:
    if base_edges_gdf.empty:
        return []
    lines_2d = []
    for geom in base_edges_gdf.geometry:
        if geom.geom_type != "LineString":
            continue
        xy = [(float(x), float(y)) for x, y, _ in geom.coords]
        if len(xy) >= 2:
            lines_2d.append(LineString(xy))
    if not lines_2d:
        return []
    polys, dangles, cuts, invalids = polygonize_full(MultiLineString(lines_2d))
    out = []
    for p in polys.geoms if hasattr(polys, "geoms") else [polys]:
        if p.is_valid and (p.area > 0.1):
            out.append(p)
    return out

def rasterize_polygons_to_mask(polys: List[Polygon], width: int, height: int, inv_affine) -> np.ndarray:
    if not polys:
        return np.zeros((height, width), dtype=np.uint8)
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img, "L")
    def worldring_to_pixels(ring):
        pts = []
        for x, y in ring.coords:
            c, r = inv_affine * (float(x), float(y))
            u = int(round(c))
            v = int(round(r))
            if -2048 <= u <= width + 2048 and -2048 <= v <= height + 2048:
                pts.append((u, v))
        return pts
    for poly in polys:
        ext = worldring_to_pixels(poly.exterior)
        if len(ext) >= 3:
            draw.polygon(ext, fill=255)
        for hole in poly.interiors:
            inn = worldring_to_pixels(hole)
            if len(inn) >= 3:
                draw.polygon(inn, fill=0)
    return np.array(mask_img, dtype=np.uint8)
