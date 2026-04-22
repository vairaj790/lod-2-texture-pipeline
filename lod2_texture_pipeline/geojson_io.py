# -*- coding: utf-8 -*-
"""GeoJSON loading and loop grouping helpers."""

from typing import Any, Dict, List
from collections import defaultdict

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
def load_3d_geojson(path):
    """
    Loads nodes/edges for convenience (legacy fields), but keeps the GDF with properties:
    - 'type' is one of {'roof','base','wall','wall_center'}
    - May include 'component_id','loop_id','ring_order' on 'base','roof','wall'.
    """
    gdf = gpd.read_file(path)
    coords = {}
    edges = defaultdict(list)
    wall_centers = []
    base_heights = []
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
    base_z = float(np.mean(base_heights)) if base_heights else 0.0
    node_ids_sorted = sorted(coords)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids_sorted)}
    corners   = np.array([coords[nid] for nid in node_ids_sorted], dtype=float)
    return gdf, corners, edges, id_to_idx, wall_centers, base_z

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
