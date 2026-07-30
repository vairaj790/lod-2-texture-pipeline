# -*- coding: utf-8 -*-
"""Run the production OSM-occlusion logic as a standalone diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lod2_texture_pipeline.osm_occlusion import (  # noqa: E402
    DEFAULT_OVERPASS_ENDPOINT,
    build_model_occlusion_geometry,
    build_osm_blocker_meshes,
    candidate_camera_pose,
    evaluate_candidate_occlusion,
    fetch_osm_buildings,
    fit_candidate_depth_global_alignment,
    mask_outline,
    parse_overpass_buildings,
    remove_target_osm_buildings,
    select_candidate_with_osm_visibility,
    target_wall_meshes,
)
from lod2_texture_pipeline.config import API_KEY, SOURCE_CRS, SV_SIZE  # noqa: E402
from lod2_texture_pipeline.projection import render_model_depth_map  # noqa: E402
from lod2_texture_pipeline.streetview import fetch_sv_image_by_id  # noqa: E402


DEFAULT_GEOJSON = (
    PROJECT_ROOT / "sample_data" / "3d_geojsons" / "building_48959353_3d.geojson"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing Street View candidates using external OSM-building "
            "depth. This diagnostic does not modify production outputs."
        )
    )
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument(
        "--production-output",
        type=Path,
        help="Existing outputs/<building> directory containing wall_artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Diagnostic result directory (defaults under this diagnostic/results).",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Process only metadata paths containing this text; repeat as needed.",
    )
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--radius-m", type=float, default=120.0)
    parser.add_argument("--default-height-m", type=float, default=15.0)
    parser.add_argument("--level-height-m", type=float, default=3.0)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.10)
    parser.add_argument("--corridor-buffer-m", type=float, default=1.0)
    parser.add_argument(
        "--clear-threshold",
        type=float,
        default=0.005,
        help="Maximum externally occluded target-wall fraction treated as clear.",
    )
    parser.add_argument("--image-size", default=SV_SIZE)
    parser.add_argument("--overpass-endpoint", default=DEFAULT_OVERPASS_ENDPOINT)
    parser.add_argument("--osm-json", type=Path, help="Use a saved Overpass JSON response.")
    parser.add_argument("--refresh-osm", action="store_true")
    return parser.parse_args()


def _safe_name(value: object) -> str:
    text = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))
    return text.strip("_") or "group"


def _read_metadata_files(
    production_output: Path,
    filters: Sequence[str],
    limit: int,
) -> List[Path]:
    files = sorted(production_output.rglob("*__ortho_meta.json"))
    if not files:
        files = sorted(production_output.rglob("*_ortho.json"))
    if filters:
        lowered = [value.lower() for value in filters]
        files = [
            path for path in files
            if any(value in str(path).lower() for value in lowered)
        ]
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(
            f"No facade metadata files found below {production_output}."
        )
    return files


def _load_source_image(candidate: Mapping[str, object], image_size: str) -> Tuple[Image.Image, str | None]:
    try:
        image, _url, _raw, _content_type = fetch_sv_image_by_id(
            str(candidate["pano_id"]),
            float(candidate["heading_deg"]),
            float(candidate["pitch_deg"]),
            float(candidate.get("fov_deg", 100.0)),
            API_KEY,
            size=image_size,
        )
        return image.convert("RGB"), None
    except Exception as exc:
        width, height = [int(value) for value in image_size.lower().split("x")]
        placeholder = Image.new("RGB", (width, height), (35, 35, 35))
        draw = ImageDraw.Draw(placeholder)
        draw.text((16, 16), "Street View image unavailable", fill=(255, 255, 255))
        draw.text((16, 40), str(exc)[:100], fill=(255, 180, 180))
        return placeholder, f"{type(exc).__name__}: {exc}"


def _draw_text_with_shadow(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    scale: float = 0.52,
) -> None:
    cv2.putText(
        image,
        text,
        (origin[0] + 1, origin[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _candidate_overlay(
    source_image: Image.Image,
    evaluation: Mapping[str, object],
    *,
    selected: bool,
    production_selected: bool,
    clear_threshold: float,
) -> Image.Image:
    canvas = np.asarray(source_image.convert("RGB"), dtype=np.uint8).copy()
    raw_target_mask = np.asarray(evaluation["raw_target_mask"], dtype=bool)
    target_mask = np.asarray(evaluation["target_mask"], dtype=bool)
    occlusion_mask = np.asarray(evaluation["occlusion_mask"], dtype=bool)
    if canvas.shape[:2] != target_mask.shape:
        canvas = cv2.resize(
            canvas,
            (target_mask.shape[1], target_mask.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    if occlusion_mask.any():
        tint = np.zeros_like(canvas)
        tint[:, :] = (255, 52, 32)
        canvas[occlusion_mask] = np.round(
            0.50 * canvas[occlusion_mask].astype(np.float32)
            + 0.50 * tint[occlusion_mask].astype(np.float32)
        ).astype(np.uint8)
    cv2.drawContours(canvas, mask_outline(raw_target_mask), -1, (0, 235, 255), 1, cv2.LINE_AA)
    cv2.drawContours(canvas, mask_outline(target_mask), -1, (255, 0, 255), 2, cv2.LINE_AA)
    if occlusion_mask.any():
        cv2.drawContours(canvas, mask_outline(occlusion_mask), -1, (255, 110, 0), 2, cv2.LINE_AA)

    candidate = evaluation["candidate"]
    fraction = float(evaluation["osm_occluded_fraction"])
    status = "CLEAR" if fraction <= float(clear_threshold) else "OBSTRUCTED"
    source_index = int(candidate.get("source_index", -1))
    rank = int(candidate.get("source_selection_rank", -1))
    _draw_text_with_shadow(
        canvas,
        f"source {source_index:02d} | production rank {rank} | OSM blocked {fraction * 100.0:.2f}% | {status}",
        (10, 23),
    )
    alignment = dict(evaluation.get("target_alignment", {}))
    if alignment.get("applied"):
        alignment_text = (
            "cyan raw | magenta corrected depth-global | orange blocked | "
            f"gain {float(alignment.get('score_improvement', 0.0)):.3f}"
        )
    else:
        alignment_text = (
            "cyan/magenta raw fallback | orange blocked | "
            f"{str(alignment.get('reason', 'fit unavailable'))[:55]}"
        )
    _draw_text_with_shadow(canvas, alignment_text, (10, 46), scale=0.43)
    if production_selected:
        _draw_text_with_shadow(canvas, "production selection", (10, 68), scale=0.46)
    if selected:
        height, width = canvas.shape[:2]
        color = (30, 230, 80)
        cv2.line(canvas, (width - 54, 30), (width - 39, 46), color, 6, cv2.LINE_AA)
        cv2.line(canvas, (width - 39, 46), (width - 13, 13), color, 6, cv2.LINE_AA)
        _draw_text_with_shadow(canvas, "OSM selection", (10, 90), scale=0.46)
    return Image.fromarray(canvas, mode="RGB")


def _polygon_pixels(polygon, to_pixel) -> np.ndarray:
    return np.asarray([to_pixel(x, y) for x, y in polygon.exterior.coords], dtype=np.int32)


def _context_map(
    model_footprint,
    blocker_lookup: Mapping[str, object],
    evaluations: Sequence[Mapping[str, object]],
    selected_evaluation: Mapping[str, object],
) -> Image.Image:
    relevant_names = {
        name
        for evaluation in evaluations
        for name in evaluation.get("candidate_blocker_mesh_names", [])
    }
    relevant_buildings = [
        blocker_lookup[name]
        for name in sorted(relevant_names)
        if name in blocker_lookup
    ]
    camera_points = [
        np.asarray(evaluation["candidate"]["camera_utm_xyz"], dtype=np.float64)[:2]
        for evaluation in evaluations
    ]
    coordinate_sets = [np.asarray(model_footprint.exterior.coords, dtype=np.float64)]
    coordinate_sets.extend(
        np.asarray(building.footprint.exterior.coords, dtype=np.float64)
        for building in relevant_buildings
    )
    if camera_points:
        coordinate_sets.append(np.asarray(camera_points, dtype=np.float64))
    all_points = np.vstack(coordinate_sets)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    center = 0.5 * (min_xy + max_xy)
    span = max(float(np.max(max_xy - min_xy)), 10.0) * 1.18
    canvas_size = 800

    def to_pixel(x, y):
        px = int(round((float(x) - (center[0] - span / 2.0)) / span * (canvas_size - 1)))
        py = int(round(((center[1] + span / 2.0) - float(y)) / span * (canvas_size - 1)))
        return px, py

    canvas = np.full((canvas_size, canvas_size, 3), 248, dtype=np.uint8)
    for building in relevant_buildings:
        points = _polygon_pixels(building.footprint, to_pixel)
        cv2.fillPoly(canvas, [points], (205, 205, 205), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [points], True, (115, 115, 115), 2, cv2.LINE_AA)
    target_points = _polygon_pixels(model_footprint, to_pixel)
    cv2.fillPoly(canvas, [target_points], (185, 235, 195), lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [target_points], True, (20, 125, 45), 3, cv2.LINE_AA)

    target_center = to_pixel(model_footprint.centroid.x, model_footprint.centroid.y)
    selected_source = int(selected_evaluation["candidate"].get("source_index", -1))
    for evaluation in evaluations:
        candidate = evaluation["candidate"]
        source_index = int(candidate.get("source_index", -1))
        camera = np.asarray(candidate["camera_utm_xyz"], dtype=np.float64)
        camera_pixel = to_pixel(camera[0], camera[1])
        is_selected = source_index == selected_source
        ray_color = (230, 170, 30) if is_selected else (80, 150, 210)
        cv2.line(canvas, camera_pixel, target_center, ray_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, camera_pixel, 8 if is_selected else 6, ray_color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(source_index),
            (camera_pixel[0] + 9, camera_pixel[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas_size - 1, 58), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "OSM context: green target | gray external buildings | yellow selected camera",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Map data: OpenStreetMap contributors",
        (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    return Image.fromarray(canvas, mode="RGB")


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def _save_gated_source(path: Path, source: Image.Image, remove_mask: np.ndarray) -> None:
    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    if rgba.shape[:2] != remove_mask.shape:
        rgba = cv2.resize(
            rgba,
            (remove_mask.shape[1], remove_mask.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    rgba[:, :, 3] = np.where(remove_mask, 0, 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _prepare_contact_sheet_image(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema() == (255, 255):
        return rgba.convert("RGB")

    width, height = rgba.size
    checker = Image.new("RGB", (width, height), "#eeeeee")
    checker_draw = ImageDraw.Draw(checker)
    cell_size = 14
    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            if ((x // cell_size) + (y // cell_size)) % 2:
                checker_draw.rectangle(
                    (x, y, min(x + cell_size - 1, width - 1), min(y + cell_size - 1, height - 1)),
                    fill="#d5d5d5",
                )
    checker.paste(rgba, (0, 0), rgba)
    return checker


def _contact_sheet(cards: Sequence[Tuple[str, Path]], destination: Path) -> None:
    if not cards:
        return
    columns = 3
    card_width, card_height = 500, 430
    rows = (len(cards) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * card_width, rows * card_height), "#f4f4f2")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (title, path) in enumerate(cards):
        column = index % columns
        row = index // columns
        x0 = column * card_width
        y0 = row * card_height
        draw.rectangle(
            (x0 + 8, y0 + 8, x0 + card_width - 8, y0 + card_height - 8),
            fill="white",
            outline="#c8c8c8",
            width=1,
        )
        draw.text((x0 + 20, y0 + 18), title[:74], fill="#202020", font=font)
        with Image.open(path) as raw:
            image = _prepare_contact_sheet_image(raw)
            fitted = ImageOps.contain(image, (card_width - 40, card_height - 70))
        px = x0 + (card_width - fitted.width) // 2
        py = y0 + 52 + (card_height - 66 - fitted.height) // 2
        sheet.paste(fitted, (px, py))
    sheet.save(destination)


def _candidate_report(evaluation: Mapping[str, object], selected_source_index: int) -> Dict[str, object]:
    candidate = evaluation["candidate"]
    source_index = int(candidate.get("source_index", -1))
    fields = (
        "source_index",
        "pano_id",
        "pano_lat",
        "pano_lng",
        "camera_utm_xyz",
        "heading_deg",
        "projection_heading_deg",
        "pitch_deg",
        "fov_deg",
        "source_selection_rank",
        "selected_for_processing",
        "target_self_visibility_fraction",
        "target_usable_visibility_fraction",
        "projected_coverage_fraction",
    )
    alignment = dict(evaluation.get("target_alignment", {}))
    alignment_H = np.asarray(
        alignment.get("homography", np.eye(3)),
        dtype=np.float64,
    )
    return {
        "candidate": {field: candidate.get(field) for field in fields},
        "selected_by_osm_experiment": source_index == selected_source_index,
        "osm_occluded_fraction": float(evaluation["osm_occluded_fraction"]),
        "osm_visible_fraction": float(evaluation["osm_visible_fraction"]),
        "target_pixel_count": int(evaluation["target_pixel_count"]),
        "osm_occluded_pixel_count": int(evaluation["osm_occluded_pixel_count"]),
        "target_projection": {
            "reference": "corrected_depth_global_wall_projection",
            "effective_mode": alignment.get("effective_mode"),
            "fit_applied": bool(alignment.get("applied", False)),
            "fit_source": alignment.get("alignment_source"),
            "fit_reason": alignment.get("reason"),
            "scale": float(alignment.get("scale", 1.0)),
            "rotation_deg": float(alignment.get("rotation_deg", 0.0)),
            "tx_px": float(alignment.get("tx_px", 0.0)),
            "ty_px": float(alignment.get("ty_px", 0.0)),
            "score_improvement": float(alignment.get("score_improvement", 0.0)),
            "H_raw_wall_to_corrected": alignment_H.astype(float).tolist(),
            "raw_target_pixel_count": int(np.asarray(evaluation["raw_target_mask"]).sum()),
            "corrected_target_pixel_count": int(evaluation["target_pixel_count"]),
        },
        "candidate_osm_buildings": list(evaluation["candidate_blocker_mesh_names"]),
        "street_view_fetch_error": evaluation.get("street_view_fetch_error"),
        "diagnostic_png": evaluation.get("diagnostic_png"),
    }


def _production_depth_global_alignment(
    metadata: Mapping[str, object],
    candidate: Mapping[str, object],
) -> Dict[str, object] | None:
    if not bool(candidate.get("selected_for_processing", False)):
        return None
    alignment = dict(metadata.get("facade_alignment") or {})
    if not alignment:
        alignment = dict(
            (metadata.get("parallel_model_depth_boundary_fit") or {}).get(
                "facade_alignment",
                {},
            )
        )
    if (
        str(alignment.get("effective_mode", "")).lower() != "depth_global"
        or not bool(alignment.get("depth_fit_accepted", False))
    ):
        return None
    H = np.asarray(alignment.get("H_raw_projection_to_selected"), dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return None
    depth_fit = dict(metadata.get("parallel_model_depth_boundary_fit") or {})
    raw_to_processing = np.asarray(
        depth_fit.get("raw_camera_to_processing_canvas_H", np.eye(3)),
        dtype=np.float64,
    )
    # The experiment displays native Street View candidates. Only metadata
    # expressed in native camera coordinates is compatible here.
    if raw_to_processing.shape != (3, 3) or not np.allclose(
        raw_to_processing,
        np.eye(3),
        atol=1.0e-8,
    ):
        return None
    return {
        "homography": H,
        "applied": True,
        "alignment_source": "accepted_production_metadata",
        "reason": str(depth_fit.get("reason", "accepted_production_depth_global_fit")),
        "transform": {
            "scale": depth_fit.get("scale", 1.0),
            "rotation_deg": depth_fit.get("rotation_deg", 0.0),
            "tx": depth_fit.get("tx_px", 0.0),
            "ty": depth_fit.get("ty_px", 0.0),
        },
        "score_improvement": depth_fit.get("score_improvement", 0.0),
        "fit_geometry_source": depth_fit.get("fit_geometry_source"),
    }


def _process_group(
    *,
    metadata_path: Path,
    output_root: Path,
    model_geometry: Mapping[str, object],
    blocker_meshes,
    blocker_lookup,
    args: argparse.Namespace,
) -> Dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    wall_indices = [int(value) for value in metadata.get("wall_global_indices", [])]
    target_mesh_list, target_quads = target_wall_meshes(model_geometry, wall_indices)
    candidates = list(metadata.get("source_candidates") or [])
    candidates = [
        candidate for candidate in candidates
        if candidate.get("pano_id") and candidate.get("camera_utm_xyz")
    ]
    if not candidates:
        raise ValueError(f"No source candidates in {metadata_path}")
    candidates.sort(key=lambda row: (
        int(row.get("source_selection_rank", 10**6)),
        int(row.get("source_index", 10**6)),
    ))

    evaluations = []
    for candidate in candidates:
        source_image, fetch_error = _load_source_image(candidate, args.image_size)
        K, R_wc, C = candidate_camera_pose(candidate, args.image_size)
        width, height = [int(value) for value in args.image_size.lower().split("x")]
        raw_target_depth = render_model_depth_map(
            target_mesh_list,
            K,
            R_wc,
            C,
            (width, height),
        )
        existing_alignment = _production_depth_global_alignment(metadata, candidate)
        if fetch_error is None or existing_alignment is not None:
            target_alignment = fit_candidate_depth_global_alignment(
                candidate=candidate,
                source_image_rgb=np.asarray(source_image.convert("RGB"), dtype=np.uint8),
                raw_target_depth=raw_target_depth,
                full_model_meshes=model_geometry["full_model_meshes"],
                model_boundary_edges_xyz=model_geometry["model_boundary_edges_xyz"],
                image_size=args.image_size,
                existing_alignment=existing_alignment,
            )
        else:
            target_alignment = {
                "homography": np.eye(3, dtype=np.float64),
                "applied": False,
                "effective_mode": "raw_fallback",
                "alignment_source": "street_view_unavailable",
                "reason": "candidate image unavailable for global-depth fitting",
                "scale": 1.0,
                "rotation_deg": 0.0,
                "tx_px": 0.0,
                "ty_px": 0.0,
                "score_improvement": 0.0,
            }
        evaluation = evaluate_candidate_occlusion(
            candidate=candidate,
            target_meshes=target_mesh_list,
            target_quads=target_quads,
            blocker_meshes=blocker_meshes,
            blocker_lookup=blocker_lookup,
            image_size=args.image_size,
            depth_tolerance_m=args.depth_tolerance_m,
            corridor_buffer_m=args.corridor_buffer_m,
            target_alignment_H=target_alignment["homography"],
            precomputed_raw_target_depth=raw_target_depth,
        )
        evaluation["candidate"] = candidate
        evaluation["target_alignment"] = target_alignment
        evaluation["street_view_fetch_error"] = fetch_error
        evaluation["_source_image"] = source_image
        evaluation["osm_fully_clear"] = bool(
            float(evaluation["osm_occluded_fraction"]) <= args.clear_threshold
        )
        evaluations.append(evaluation)

    selection = select_candidate_with_osm_visibility(
        evaluations,
        clear_occlusion_fraction=args.clear_threshold,
    )
    selected_evaluation = selection["selected"]
    selected_candidate = selected_evaluation["candidate"]
    selected_source_index = int(selected_candidate.get("source_index", -1))
    facade_tag = metadata.get("facade_tag") or metadata_path.stem.replace("__ortho_meta", "")
    group_directory = output_root / _safe_name(facade_tag)
    group_directory.mkdir(parents=True, exist_ok=True)
    cards: List[Tuple[str, Path]] = []

    context_path = group_directory / "00_osm_camera_context.png"
    _context_map(
        model_geometry["footprint"],
        blocker_lookup,
        evaluations,
        selected_evaluation,
    ).save(context_path)
    cards.append(("00 OSM camera-to-wall context", context_path))

    selected_source_image = None
    for evaluation in evaluations:
        candidate = evaluation["candidate"]
        source_index = int(candidate.get("source_index", -1))
        source_image = evaluation["_source_image"]
        is_selected = source_index == selected_source_index
        if is_selected:
            selected_source_image = source_image
        overlay = _candidate_overlay(
            source_image,
            evaluation,
            selected=is_selected,
            production_selected=bool(candidate.get("selected_for_processing", False)),
            clear_threshold=args.clear_threshold,
        )
        suffix = "_SELECTED" if is_selected else ""
        overlay_path = group_directory / f"01_candidate_{source_index:02d}_osm_occlusion{suffix}.png"
        overlay.save(overlay_path)
        evaluation["diagnostic_png"] = str(overlay_path)
        cards.append((f"01 candidate source {source_index:02d}{suffix}", overlay_path))

    selected_mask = np.asarray(selected_evaluation["occlusion_mask"], dtype=bool)
    mask_path = group_directory / "02_selected_external_building_removal_mask.png"
    _save_mask(mask_path, selected_mask)
    cards.append(("02 selected external-building removal mask", mask_path))

    gated_path = None
    if selection["fallback_mask_required"]:
        if selected_source_image is None:
            selected_source_image, _error = _load_source_image(selected_candidate, args.image_size)
        gated_path = group_directory / "03_selected_source_external_buildings_removed.png"
        _save_gated_source(gated_path, selected_source_image, selected_mask)
        cards.append(("03 fallback source with OSM obstruction removed (checkerboard = removed)", gated_path))

    contact_sheet_path = group_directory / "contact_sheet.png"
    _contact_sheet(cards, contact_sheet_path)
    candidate_reports = [
        _candidate_report(evaluation, selected_source_index)
        for evaluation in evaluations
    ]
    group_report = {
        "facade_tag": facade_tag,
        "production_metadata": str(metadata_path),
        "occlusion_target_projection": "corrected_depth_global_wall_projection",
        "wall_global_indices": wall_indices,
        "production_selected_source_index": next(
            (
                int(candidate.get("source_index", -1))
                for candidate in candidates
                if candidate.get("selected_for_processing")
            ),
            None,
        ),
        "osm_selected_source_index": selected_source_index,
        "selection_reason": selection["selection_reason"],
        "fallback_mask_required": bool(selection["fallback_mask_required"]),
        "clear_candidate_count": int(selection["clear_candidate_count"]),
        "usable_candidate_count": int(selection["usable_candidate_count"]),
        "clear_occlusion_fraction": float(selection["clear_occlusion_fraction"]),
        "selected_external_occlusion_fraction": float(
            selected_evaluation["osm_occluded_fraction"]
        ),
        "selected_removal_mask_png": str(mask_path),
        "selected_gated_source_png": str(gated_path) if gated_path else None,
        "context_map_png": str(context_path),
        "contact_sheet_png": str(contact_sheet_path),
        "candidates": candidate_reports,
    }
    (group_directory / "report.json").write_text(
        json.dumps(group_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return group_report


def main() -> int:
    args = _parse_args()
    geojson_path = args.geojson.resolve()
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)
    production_output = (
        args.production_output.resolve()
        if args.production_output
        else PROJECT_ROOT / "outputs" / geojson_path.stem
    )
    output_root = (
        args.output.resolve()
        if args.output
        else Path(__file__).resolve().parent / "results" / geojson_path.stem
    )
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_files = _read_metadata_files(
        production_output,
        args.group,
        args.limit_groups,
    )
    model_geometry = build_model_occlusion_geometry(geojson_path)

    if args.osm_json:
        osm_payload = json.loads(args.osm_json.read_text(encoding="utf-8"))
        osm_buildings = parse_overpass_buildings(
            osm_payload,
            target_crs=SOURCE_CRS,
            default_height_m=args.default_height_m,
            level_height_m=args.level_height_m,
        )
        osm_metadata = {
            "source": str(args.osm_json.resolve()),
            "parsed_building_count": len(osm_buildings),
            "attribution": "OpenStreetMap contributors",
            "license": "ODbL 1.0",
        }
    else:
        osm_buildings, osm_metadata = fetch_osm_buildings(
            model_footprint=model_geometry["footprint"],
            source_crs=SOURCE_CRS,
            radius_m=args.radius_m,
            endpoint=args.overpass_endpoint,
            cache_dir=Path(__file__).resolve().parent / "cache",
            refresh=args.refresh_osm,
            default_height_m=args.default_height_m,
            level_height_m=args.level_height_m,
        )
    blocker_buildings, excluded_target_keys = remove_target_osm_buildings(
        osm_buildings,
        model_geometry["footprint"],
    )
    blocker_meshes, blocker_lookup = build_osm_blocker_meshes(
        blocker_buildings,
        ground_z=float(model_geometry["base_z"]),
    )

    print(
        f"[OSM experiment] {len(metadata_files)} groups, "
        f"{len(osm_buildings)} OSM buildings, {len(blocker_meshes)} external blockers"
    )
    group_reports = []
    failures = []
    for index, metadata_path in enumerate(metadata_files, start=1):
        try:
            report = _process_group(
                metadata_path=metadata_path,
                output_root=output_root,
                model_geometry=model_geometry,
                blocker_meshes=blocker_meshes,
                blocker_lookup=blocker_lookup,
                args=args,
            )
            group_reports.append(report)
            print(
                f"[{index}/{len(metadata_files)}] {report['facade_tag']}: "
                f"source {report['osm_selected_source_index']} "
                f"({report['selected_external_occlusion_fraction'] * 100.0:.2f}% blocked)"
            )
        except Exception as exc:
            failures.append({
                "metadata": str(metadata_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{index}/{len(metadata_files)}] FAILED {metadata_path.name}: {exc}")

    summary = {
        "experiment": "osm_building_occlusion",
        "production_pipeline_modified": False,
        "occlusion_target_projection": "corrected_depth_global_wall_projection",
        "geojson": str(geojson_path),
        "production_output": str(production_output),
        "source_crs": SOURCE_CRS,
        "parameters": {
            "radius_m": args.radius_m,
            "default_height_m": args.default_height_m,
            "level_height_m": args.level_height_m,
            "depth_tolerance_m": args.depth_tolerance_m,
            "corridor_buffer_m": args.corridor_buffer_m,
            "clear_threshold": args.clear_threshold,
            "image_size": args.image_size,
        },
        "osm": osm_metadata,
        "excluded_target_osm_buildings": excluded_target_keys,
        "external_blocker_count": len(blocker_meshes),
        "groups": group_reports,
        "failures": failures,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OSM experiment] summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
