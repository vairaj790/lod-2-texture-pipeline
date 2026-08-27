# -*- coding: utf-8 -*-
"""Collect batch exports and summarize wall-texturing coverage.

This is an offline report: it reads source GeoJSON and persisted pipeline
artifacts only.  It does not call Street View, SAM3, or rerun the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BUILDING_DIRECTORY = re.compile(r"^building_.+_3d$")
IMMEDIATE_SOURCE_RETURN_SECONDS = 0.01


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _optional_number_sort_key(value: Any) -> tuple[int, float, str]:
    if value is None:
        return (1, 0.0, "")
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (0, 0.0, str(value))


def _wall_descriptors(geojson_path: Path) -> list[dict[str, Any]]:
    """Recreate the pipeline's stable global wall order from wall lines."""
    document = _read_json(geojson_path)
    candidates = []
    for source_order, feature in enumerate(document.get("features", [])):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        if str(geometry.get("type", "")).lower() != "linestring":
            continue
        if str(properties.get("type", "")).lower() != "wall":
            continue
        candidates.append({
            "source_order": int(source_order),
            "component_id": properties.get("component_id"),
            "loop_id": properties.get("loop_id"),
            "loop_index": properties.get("ring_order"),
        })

    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["component_id"], row["loop_id"])].append(row)

    ordered = []
    group_keys = sorted(
        grouped,
        key=lambda key: (
            _optional_number_sort_key(key[0]),
            _optional_number_sort_key(key[1]),
        ),
    )
    for key in group_keys:
        group = grouped[key]
        group.sort(
            key=lambda row: (
                _optional_number_sort_key(row["loop_index"]),
                row["source_order"],
            )
        )
        ordered.extend(group)

    for global_index, row in enumerate(ordered):
        row["global_wall_index"] = int(global_index)
    return ordered


def _matching_geojson_directory(
    workspace: Path,
    building_names: set[str],
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        directory = explicit.resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"GeoJSON directory does not exist: {directory}")
        return directory

    candidates = [
        workspace / "sample_data" / "3d_geojsons",
        workspace / "raw_data" / "3d_geojsons",
        workspace.parent / "raw_data" / "3d_geojsons",
    ]
    scored = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        names = {path.stem for path in directory.glob("*.geojson")}
        scored.append((len(names & building_names), directory.resolve()))
    if not scored:
        raise FileNotFoundError(
            "Could not find a GeoJSON directory. Pass --geojson-dir explicitly."
        )
    matched, selected = max(scored, key=lambda item: item[0])
    if matched != len(building_names):
        raise RuntimeError(
            f"Best GeoJSON directory matches {matched}/{len(building_names)} "
            f"building folders: {selected}"
        )
    return selected


def _stage_information(building_directory: Path) -> dict[str, Any]:
    timing_path = building_directory / "stage_timings.json"
    if not timing_path.is_file():
        raise FileNotFoundError(f"Missing stage timings: {timing_path}")
    timing = _read_json(timing_path)
    events = list(timing.get("events", []))
    stages = [str(event.get("stage", "")) for event in events]
    fetch_seconds = {}
    suffix = " / fetch SV + source selection"
    for event in events:
        stage = str(event.get("stage", ""))
        if stage.endswith(suffix):
            fetch_seconds[stage[: -len(suffix)]] = float(event.get("seconds", 0.0))
    no_panorama = bool(
        stages
        and stages[-1] == "Street View pano discovery"
        and "build placeholder meshes and wall records" not in stages
    )
    return {
        "path": timing_path,
        "event_count": len(events),
        "last_stage": stages[-1] if stages else None,
        "no_panorama": no_panorama,
        "fetch_seconds": fetch_seconds,
        "total_seconds": float(timing.get("total_seconds", 0.0)),
    }


def _single_export(building_directory: Path, suffix: str) -> Path | None:
    matches = sorted(building_directory.glob(f"*__textured.{suffix}"))
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected at most one {suffix.upper()} in {building_directory}, "
            f"found {len(matches)}."
        )
    return matches[0] if matches else None


def _processed_wall_rows(
    building_directory: Path,
    total_walls: int,
    stage: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[int]]:
    summaries = sorted(
        (building_directory / "wall_artifacts").glob("group_*/group_summary.json")
    )
    rows = []
    seen = set()
    for summary_path in summaries:
        summary = _read_json(summary_path)
        group_directory = summary_path.parent
        for raw in summary.get("rows", []):
            global_index = int(raw["global_index"])
            if global_index in seen:
                raise RuntimeError(
                    f"Duplicate wall {global_index} in {building_directory.name}."
                )
            seen.add(global_index)
            ortho_name = raw.get("ortho_png")
            texture_exists = bool(
                ortho_name and (group_directory / Path(str(ortho_name)).name).is_file()
            )
            textured = bool(
                not bool(raw.get("debug_only", False))
                and ortho_name
                and texture_exists
            )
            facade_tag = str(raw.get("facade_group_tag") or "")
            source_seconds = stage["fetch_seconds"].get(facade_tag)
            immediate_hint = bool(
                not textured
                and source_seconds is not None
                and source_seconds <= IMMEDIATE_SOURCE_RETURN_SECONDS
            )
            rows.append({
                "building_id": building_directory.name,
                "global_wall_index": global_index,
                "component_id": raw.get("component_id"),
                "loop_id": raw.get("loop_id"),
                "loop_index": raw.get("loop_index"),
                "facade_group_id": raw.get("facade_group_id"),
                "facade_group_tag": facade_tag or None,
                "wall_status": "textured" if textured else "untextured",
                "untextured_reason": (
                    None if textured else "source_failure_reason_not_recorded"
                ),
                "reason_evidence": (
                    "ortho_png_exists_and_group_row_is_not_debug_only"
                    if textured
                    else "generic_placeholder_group_summary"
                ),
                "non_authoritative_reason_hint": (
                    "likely_no_candidate_image_immediate_source_return"
                    if immediate_hint else None
                ),
                "source_selection_seconds": source_seconds,
                "ortho_png": Path(str(ortho_name)).name if ortho_name else None,
                "group_summary": str(summary_path.resolve()),
            })

    if len(seen) != total_walls or seen != set(range(total_walls)):
        raise RuntimeError(
            f"Wall inventory mismatch for {building_directory.name}: "
            f"GeoJSON={total_walls}, group summaries={len(seen)}."
        )
    return sorted(rows, key=lambda row: row["global_wall_index"]), seen


def _early_exit_wall_rows(
    building_directory: Path,
    descriptors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "building_id": building_directory.name,
            "global_wall_index": row["global_wall_index"],
            "component_id": row.get("component_id"),
            "loop_id": row.get("loop_id"),
            "loop_index": row.get("loop_index"),
            "facade_group_id": None,
            "facade_group_tag": None,
            "wall_status": "untextured",
            "untextured_reason": "no_image_no_panorama_discovered",
            "reason_evidence": "stage_timings_ended_at_street_view_pano_discovery",
            "non_authoritative_reason_hint": None,
            "source_selection_seconds": None,
            "ortho_png": None,
            "group_summary": None,
        }
        for row in descriptors
    ]


def analyze_building(
    building_directory: Path,
    geojson_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path | None]]:
    descriptors = _wall_descriptors(geojson_path)
    total_walls = len(descriptors)
    if total_walls <= 0:
        raise RuntimeError(f"No wall LineStrings found in {geojson_path}")
    stage = _stage_information(building_directory)
    viewer_path = building_directory / "viewer_index.json"
    if stage["no_panorama"]:
        wall_rows = _early_exit_wall_rows(building_directory, descriptors)
        if viewer_path.exists():
            raise RuntimeError(
                f"No-panorama run unexpectedly has viewer_index: {viewer_path}"
            )
    else:
        wall_rows, _ = _processed_wall_rows(
            building_directory, total_walls, stage
        )
        if not viewer_path.is_file():
            raise FileNotFoundError(
                f"Processed building is missing viewer_index: {viewer_path}"
            )
        viewer_rows = list(_read_json(viewer_path))
        viewer_indices = {int(row["global_index"]) for row in viewer_rows}
        textured_indices = {
            int(row["global_wall_index"])
            for row in wall_rows if row["wall_status"] == "textured"
        }
        if viewer_indices != textured_indices:
            raise RuntimeError(
                f"viewer_index/group-summary mismatch for {building_directory.name}: "
                f"viewer={len(viewer_indices)}, textured={len(textured_indices)}"
            )

    textured = sum(row["wall_status"] == "textured" for row in wall_rows)
    untextured = total_walls - textured
    no_image = sum(
        row["untextured_reason"] == "no_image_no_panorama_discovered"
        for row in wall_rows
    )
    unknown = sum(
        row["untextured_reason"] == "source_failure_reason_not_recorded"
        for row in wall_rows
    )
    timing_hint = sum(
        bool(row["non_authoritative_reason_hint"]) for row in wall_rows
    )
    if textured == total_walls:
        status = "fully_textured"
    elif textured > 0:
        status = "partially_textured"
    elif stage["no_panorama"]:
        status = "untextured_no_panorama"
    else:
        status = "untextured_after_panorama_discovery"

    kmz = _single_export(building_directory, "kmz")
    glb = _single_export(building_directory, "glb")
    building_row = {
        "building_id": building_directory.name,
        "pipeline_status": status,
        "total_walls": total_walls,
        "textured_walls": textured,
        "untextured_walls": untextured,
        "texturing_percent": round(100.0 * textured / total_walls, 6),
        "has_any_textured_wall": textured > 0,
        "fully_textured": textured == total_walls,
        "no_panorama_discovered": stage["no_panorama"],
        "untextured_no_image_confirmed": no_image,
        # The current artifacts do not persist a whole-facade occlusion
        # rejection reason, so this must remain zero rather than be guessed.
        "untextured_occlusion_confirmed": 0,
        "untextured_source_failure_reason_not_recorded": unknown,
        "reason_not_recorded_likely_no_candidate_timing_hint": timing_hint,
        "kmz_exported": kmz is not None,
        "glb_exported": glb is not None,
        "pipeline_total_seconds": round(stage["total_seconds"], 6),
        "last_recorded_stage": stage["last_stage"],
        "geojson_path": str(geojson_path.resolve()),
        "viewer_index_path": str(viewer_path.resolve()) if viewer_path.exists() else None,
    }
    return building_row, wall_rows, {"kmz": kmz, "glb": glb}


def _copy_exports(
    export_rows: list[tuple[str, dict[str, Path | None]]],
    outputs_directory: Path,
) -> list[dict[str, Any]]:
    destinations = {
        "kmz": outputs_directory / "kmz_files",
        "glb": outputs_directory / "glb_files",
    }
    for destination in destinations.values():
        destination.mkdir(parents=True, exist_ok=True)

    names_by_extension: dict[str, set[str]] = defaultdict(set)
    manifest = []
    for building_id, exports in export_rows:
        for extension in ("kmz", "glb"):
            source = exports.get(extension)
            if source is None:
                continue
            if source.name in names_by_extension[extension]:
                raise RuntimeError(
                    f"Duplicate {extension.upper()} basename: {source.name}"
                )
            names_by_extension[extension].add(source.name)
            destination = destinations[extension] / source.name
            shutil.copy2(source, destination)
            source_size = int(source.stat().st_size)
            copied_size = int(destination.stat().st_size)
            if copied_size != source_size:
                raise IOError(
                    f"Copied file size mismatch: {source} -> {destination}"
                )
            manifest.append({
                "building_id": building_id,
                "file_type": extension,
                "source_path": str(source.resolve()),
                "destination_path": str(destination.resolve()),
                "size_bytes": source_size,
                "copy_verified_by_size": True,
            })
    return manifest


def _build_summary(
    building_rows: list[dict[str, Any]],
    wall_rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    outputs_directory: Path,
    geojson_directory: Path,
) -> dict[str, Any]:
    building_count = len(building_rows)
    total_walls = sum(int(row["total_walls"]) for row in building_rows)
    textured_walls = sum(int(row["textured_walls"]) for row in building_rows)
    untextured_walls = total_walls - textured_walls
    percentages = [float(row["texturing_percent"]) for row in building_rows]
    statuses = defaultdict(int)
    for row in building_rows:
        statuses[str(row["pipeline_status"])] += 1
    no_image = sum(
        int(row["untextured_no_image_confirmed"]) for row in building_rows
    )
    unknown = sum(
        int(row["untextured_source_failure_reason_not_recorded"])
        for row in building_rows
    )
    inferred_no_candidate_walls = sum(
        int(row["reason_not_recorded_likely_no_candidate_timing_hint"])
        for row in building_rows
    )
    inferred_groups = {
        (row["building_id"], row["facade_group_tag"])
        for row in wall_rows
        if row["non_authoritative_reason_hint"]
    }
    copied_by_type = defaultdict(list)
    for row in manifest:
        copied_by_type[str(row["file_type"])].append(row)
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "offline_analysis_only": True,
        "pipeline_rerun": False,
        "paths": {
            "outputs_directory": str(outputs_directory.resolve()),
            "geojson_directory": str(geojson_directory.resolve()),
            "kmz_collection": str((outputs_directory / "kmz_files").resolve()),
            "glb_collection": str((outputs_directory / "glb_files").resolve()),
        },
        "buildings": {
            "total": building_count,
            "with_any_textured_wall": sum(
                bool(row["has_any_textured_wall"]) for row in building_rows
            ),
            "fully_textured": statuses["fully_textured"],
            "partially_textured": statuses["partially_textured"],
            "zero_textured": (
                statuses["untextured_no_panorama"]
                + statuses["untextured_after_panorama_discovery"]
            ),
            "no_panorama_discovered": statuses["untextured_no_panorama"],
            "zero_after_panorama_discovery": statuses[
                "untextured_after_panorama_discovery"
            ],
            "mean_texturing_percent": statistics.mean(percentages),
            "median_texturing_percent": statistics.median(percentages),
        },
        "walls": {
            "total": total_walls,
            "textured": textured_walls,
            "untextured": untextured_walls,
            "weighted_texturing_percent": (
                100.0 * textured_walls / total_walls if total_walls else 0.0
            ),
            "processed_building_total": total_walls - no_image,
            "processed_building_textured": textured_walls,
            "processed_building_texturing_percent": (
                100.0 * textured_walls / (total_walls - no_image)
                if total_walls > no_image else 0.0
            ),
        },
        "untextured_reason_matrix": {
            "no_image_confirmed": no_image,
            "occlusion_confirmed": 0,
            "source_failure_reason_not_recorded": unknown,
            "total": untextured_walls,
        },
        "non_authoritative_timing_hint": {
            "description": (
                "Subset of source_failure_reason_not_recorded whose source-selection "
                "stage returned in <= 0.01 seconds; code flow strongly suggests no "
                "candidate panorama, but the failure reason was not persisted."
            ),
            "groups": len(inferred_groups),
            "walls": inferred_no_candidate_walls,
        },
        "exports": {
            "kmz_copied": len(copied_by_type["kmz"]),
            "glb_copied": len(copied_by_type["glb"]),
            "kmz_bytes": sum(int(row["size_bytes"]) for row in copied_by_type["kmz"]),
            "glb_bytes": sum(int(row["size_bytes"]) for row in copied_by_type["glb"]),
        },
        "data_quality": {
            "total_wall_source": "source GeoJSON wall LineString features",
            "textured_wall_rule": (
                "group row is not debug-only and its referenced ortho PNG exists"
            ),
            "viewer_index_cross_check": "required exact global-index match",
            "reason_limitation": (
                "Untextured facade groups after panorama discovery persist only a "
                "generic placeholder status. No-image, fetch failure, invalid/edge-on "
                "projection, and semantic/OSM occlusion cannot be separated exactly."
            ),
            "occlusion_zero_interpretation": (
                "Zero means no whole-facade occlusion rejection was explicitly stored; "
                "it does not mean the run contained no occlusions."
            ),
        },
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    buildings = summary["buildings"]
    walls = summary["walls"]
    reasons = summary["untextured_reason_matrix"]
    exports = summary["exports"]
    hint = summary["non_authoritative_timing_hint"]
    return f"""# Batch texture statistics

This report was generated from persisted output JSON/artifacts and the source
GeoJSON wall inventory. The texture pipeline was **not rerun**.

## Building results

| Result | Buildings |
|---|---:|
| Total | {buildings['total']} |
| At least one textured wall | {buildings['with_any_textured_wall']} |
| Fully textured | {buildings['fully_textured']} |
| Partially textured | {buildings['partially_textured']} |
| Zero textured walls | {buildings['zero_textured']} |
| No panoramas discovered | {buildings['no_panorama_discovered']} |
| Zero texture after panorama discovery | {buildings['zero_after_panorama_discovery']} |

Mean per-building coverage: **{buildings['mean_texturing_percent']:.3f}%**
Median per-building coverage: **{buildings['median_texturing_percent']:.3f}%**

## Wall results

| Result | Walls | Percent |
|---|---:|---:|
| Total | {walls['total']} | 100.000% |
| Textured | {walls['textured']} | {walls['weighted_texturing_percent']:.3f}% |
| Untextured | {walls['untextured']} | {100.0 - walls['weighted_texturing_percent']:.3f}% |

Among buildings that reached facade processing, {walls['textured']} of
{walls['processed_building_total']} walls were textured
({walls['processed_building_texturing_percent']:.3f}%).

## Untextured-reason matrix

| Persisted reason | Walls |
|---|---:|
| No image: no panoramas discovered | {reasons['no_image_confirmed']} |
| Occlusion explicitly confirmed | {reasons['occlusion_confirmed']} |
| Source failure reason not recorded | {reasons['source_failure_reason_not_recorded']} |
| Total untextured | {reasons['total']} |

The zero in "occlusion explicitly confirmed" does **not** mean that no walls
were occluded. For failed facade groups, the current JSON stores only
`geometry_group_exists_no_texture_artifacts_yet`; it discards whether selection
failed because there was no usable image, complete semantic/OSM occlusion, an
invalid projection, or an edge-on view. Therefore those categories cannot be
split exactly after the run without inventing data.

As a non-authoritative diagnostic only, {hint['groups']} failed groups covering
{hint['walls']} walls returned from source selection in at most 0.01 seconds.
The code path strongly suggests "no candidate panorama," but this remains a
timing inference and is not included in the confirmed no-image total.

## Collected exports

| Folder | Files | Bytes |
|---|---:|---:|
| `outputs/kmz_files` | {exports['kmz_copied']} | {exports['kmz_bytes']} |
| `outputs/glb_files` | {exports['glb_copied']} | {exports['glb_bytes']} |

See `building_statistics.csv` for one row per building,
`untextured_reason_matrix.csv` for the requested building-level reason matrix,
`wall_statistics.csv` for one row per wall, and `copy_manifest.csv` for every
copied model.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--geojson-dir", type=Path, default=None)
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Generate statistics without copying KMZ/GLB exports.",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    outputs_directory = args.outputs_dir.resolve()
    if not outputs_directory.is_dir():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_directory}")
    building_directories = sorted(
        path
        for path in outputs_directory.iterdir()
        if path.is_dir() and BUILDING_DIRECTORY.fullmatch(path.name)
    )
    if not building_directories:
        raise RuntimeError(f"No building output directories found in {outputs_directory}")
    building_names = {path.name for path in building_directories}
    geojson_directory = _matching_geojson_directory(
        workspace, building_names, args.geojson_dir
    )

    building_rows = []
    wall_rows = []
    export_rows = []
    for building_directory in building_directories:
        geojson_path = geojson_directory / f"{building_directory.name}.geojson"
        if not geojson_path.is_file():
            raise FileNotFoundError(f"Missing source GeoJSON: {geojson_path}")
        building_row, building_walls, exports = analyze_building(
            building_directory, geojson_path
        )
        building_rows.append(building_row)
        wall_rows.extend(building_walls)
        export_rows.append((building_directory.name, exports))

    manifest = [] if args.no_copy else _copy_exports(export_rows, outputs_directory)
    report_directory = outputs_directory / "batch_statistics"
    report_directory.mkdir(parents=True, exist_ok=True)

    building_fields = [
        "building_id", "pipeline_status", "total_walls", "textured_walls",
        "untextured_walls", "texturing_percent", "has_any_textured_wall",
        "fully_textured", "no_panorama_discovered",
        "untextured_no_image_confirmed", "untextured_occlusion_confirmed",
        "untextured_source_failure_reason_not_recorded",
        "reason_not_recorded_likely_no_candidate_timing_hint",
        "kmz_exported", "glb_exported", "pipeline_total_seconds",
        "last_recorded_stage", "geojson_path", "viewer_index_path",
    ]
    wall_fields = [
        "building_id", "global_wall_index", "component_id", "loop_id",
        "loop_index", "facade_group_id", "facade_group_tag", "wall_status",
        "untextured_reason", "reason_evidence",
        "non_authoritative_reason_hint", "source_selection_seconds",
        "ortho_png", "group_summary",
    ]
    reason_rows = [
        {
            "building_id": row["building_id"],
            "total_untextured_walls": row["untextured_walls"],
            "no_image_confirmed": row["untextured_no_image_confirmed"],
            "occlusion_confirmed": row["untextured_occlusion_confirmed"],
            "source_failure_reason_not_recorded": row[
                "untextured_source_failure_reason_not_recorded"
            ],
            "not_recorded_likely_no_candidate_timing_hint": row[
                "reason_not_recorded_likely_no_candidate_timing_hint"
            ],
        }
        for row in building_rows
    ]
    reason_fields = [
        "building_id", "total_untextured_walls", "no_image_confirmed",
        "occlusion_confirmed", "source_failure_reason_not_recorded",
        "not_recorded_likely_no_candidate_timing_hint",
    ]
    manifest_fields = [
        "building_id", "file_type", "source_path", "destination_path",
        "size_bytes", "copy_verified_by_size",
    ]

    _write_csv(
        report_directory / "building_statistics.csv",
        building_rows,
        building_fields,
    )
    _write_csv(
        report_directory / "untextured_reason_matrix.csv",
        reason_rows,
        reason_fields,
    )
    _write_csv(
        report_directory / "wall_statistics.csv",
        wall_rows,
        wall_fields,
    )
    _write_csv(
        report_directory / "copy_manifest.csv",
        manifest,
        manifest_fields,
    )
    summary = _build_summary(
        building_rows,
        wall_rows,
        manifest,
        outputs_directory,
        geojson_directory,
    )
    _write_json(report_directory / "summary.json", summary)
    (report_directory / "SUMMARY.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )

    print(f"Buildings: {summary['buildings']['total']}")
    print(
        "Walls textured: "
        f"{summary['walls']['textured']}/{summary['walls']['total']} "
        f"({summary['walls']['weighted_texturing_percent']:.3f}%)"
    )
    print(
        "Untextured reasons: "
        f"no-image confirmed={summary['untextured_reason_matrix']['no_image_confirmed']}, "
        f"occlusion confirmed={summary['untextured_reason_matrix']['occlusion_confirmed']}, "
        "not recorded="
        f"{summary['untextured_reason_matrix']['source_failure_reason_not_recorded']}"
    )
    print(
        f"Copied KMZ={summary['exports']['kmz_copied']}, "
        f"GLB={summary['exports']['glb_copied']}"
    )
    print(f"Report: {report_directory}")


if __name__ == "__main__":
    main()
