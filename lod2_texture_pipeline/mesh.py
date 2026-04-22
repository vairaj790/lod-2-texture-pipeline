# -*- coding: utf-8 -*-
"""Mesh building, triangulation, roof polygonization, and raster masking."""

from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize_full
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
