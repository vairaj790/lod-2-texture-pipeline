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
