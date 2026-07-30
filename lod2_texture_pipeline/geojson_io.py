# -*- coding: utf-8 -*-
"""GeoJSON loading and loop grouping helpers."""

from typing import Any, Dict, List
from collections import defaultdict
import ast
import json

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    if type(value).__name__ == "NAType":
        return True
    try:
        return bool(value != value)
    except Exception:
        return False

def _as_int_list(value: Any) -> List[int]:
    if value is None or _is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(text)
                break
            except Exception:
                pass
        else:
            value = [p.strip() for p in text.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if item is None or _is_missing(item):
                continue
            out.append(int(item))
        return out
    return [int(value)]

def _props_from_row(row) -> Dict[str, Any]:
    props = {}
    for key, value in row.items():
        if key == "geometry" or _is_missing(value):
            continue
        props[key] = value
    return props

def load_3d_geojson(path):
    """
    Loads nodes/edges for convenience (legacy fields), but keeps the GDF with properties:
    - 'type' is one of {'roof','base','wall','wall_center'}
    - May include 'component_id','loop_id','ring_order' on 'base','roof','wall'.
    - May include explicit Polygon surface features from the reconstruction script.
    """
    # Pin the backend so the environment does not need the unused Fiona stack.
    gdf = gpd.read_file(path, engine="pyogrio")
    coords = {}
    edges = defaultdict(list)
    wall_centers = []
    base_heights = []
    surface_rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            s, t = int(row['source']), int(row['target'])
            typ  = str(row['type'])
            coords[s] = geom.coords[0]
            coords[t] = geom.coords[1]
            edges[typ].append((s, t))
            if typ == 'base':
                base_heights.extend([geom.coords[0][2], geom.coords[1][2]])
        elif (str(row.get("type", "")) == "wall_center") and (geom is not None) and (geom.geom_type == "Point"):
            wall_centers.append(np.array(geom.coords[0], dtype=float))
        elif isinstance(geom, Polygon):
            props = _props_from_row(row)
            if (
                str(props.get("feature_kind", "")).lower() == "surface"
                or props.get("surface_type") is not None
                or props.get("vertex_ids") is not None
                or props.get("vertex_indices") is not None
            ):
                surface_rows.append((geom, props))
    base_z = float(np.mean(base_heights)) if base_heights else 0.0
    node_ids_sorted = sorted(coords)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids_sorted)}
    corners   = np.array([coords[nid] for nid in node_ids_sorted], dtype=float)

    coord_to_idx = {
        tuple(np.round(np.asarray(xyz, dtype=float), 7)): idx
        for idx, xyz in enumerate(corners)
    }
    surface_faces = []
    for geom, props in surface_rows:
        vertex_ids = _as_int_list(props.get("vertex_ids"))
        ring = [id_to_idx[v] for v in vertex_ids if v in id_to_idx]

        if len(ring) < 3:
            raw_indices = _as_int_list(props.get("vertex_indices"))
            if raw_indices and all(0 <= v < len(corners) for v in raw_indices):
                ring = raw_indices

        if len(ring) < 3:
            ring = []
            for xyz in geom.exterior.coords[:-1]:
                idx = coord_to_idx.get(tuple(np.round(np.asarray(xyz, dtype=float), 7)))
                if idx is not None:
                    ring.append(idx)

        if len(ring) < 3:
            continue

        sf = dict(props)
        sf["surface_type"] = str(sf.get("surface_type", sf.get("type", ""))).lower()
        sf["vertex_indices"] = [int(v) for v in ring]
        surface_faces.append(sf)

    return gdf, corners, edges, id_to_idx, wall_centers, base_z, surface_faces

def build_edge_loops_from_gdf(gdf: "gpd.GeoDataFrame", edge_type: str) -> List[Dict[str, Any]]:
    """
    Generic loop builder for wall/base/roof edges using (component_id, loop_id, ring_order).
    Falls back to a single loop when props are absent.
    """
    if not {'type','source','target'}.issubset(set(gdf.columns)):
        df = gdf[gdf['type']==edge_type]
        return [{'component_id': None, 'loop_id': None,
                 'edges': [(int(r['source']), int(r['target'])) for _, r in df.iterrows()]}]

    df = gdf[gdf['type']==edge_type].copy()
    has_group = all(c in df.columns for c in ['component_id','loop_id'])
    has_order = 'ring_order' in df.columns

    if not has_group:
        return [{'component_id': None, 'loop_id': None,
                 'edges': [(int(r['source']), int(r['target'])) for _, r in df.iterrows()]}]

    loops = []
    for (cid, lid), d in df.groupby(['component_id','loop_id'], dropna=False, sort=True):
        d2 = d.sort_values('ring_order', kind='mergesort') if has_order else d
        edges = [(int(r['source']), int(r['target'])) for _, r in d2.iterrows()]
        if len(edges) >= 2:
            loops.append({'component_id': int(cid) if cid==cid else None,
                          'loop_id': int(lid) if lid==lid else None,
                          'edges': edges})
    return loops
