# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from lod2_texture_pipeline.config import OUTPUT_DIR, SAM3_PROMPT_FACADE, SAM3_PROMPT_FACADE_REFINEMENT, SAM3_PROMPT_ROOF


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_GEOJSON_PATH = REPO_ROOT / "sample_data" / "3d_geojsons" / "building_48959353_3d.geojson"
DEFAULT_GEOTIFF_PATH = REPO_ROOT / "sample_data" / "geotiffs" / "building_48959353.tif"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the texture pipeline for one LoD-2 building.",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=DEFAULT_GEOJSON_PATH,
        help=f"3D building GeoJSON (default: {DEFAULT_GEOJSON_PATH})",
    )
    parser.add_argument(
        "--geotiff",
        type=Path,
        default=DEFAULT_GEOTIFF_PATH,
        help="Optional roof orthophoto GeoTIFF.",
    )
    parser.add_argument(
        "--no-geotiff",
        action="store_true",
        help="Run without a roof GeoTIFF; roof faces remain untextured.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    geojson_path = args.geojson.expanduser().resolve()
    geotiff_path = None if args.no_geotiff else args.geotiff.expanduser().resolve()

    if not geojson_path.is_file():
        raise FileNotFoundError(f"GeoJSON input not found: {geojson_path}")
    if geotiff_path is not None and not geotiff_path.is_file():
        raise FileNotFoundError(
            f"GeoTIFF input not found: {geotiff_path}. "
            "Pass --no-geotiff to run without roof imagery."
        )

    # Delay heavyweight torch/SAM3 imports until after arguments and inputs
    # have been validated. This also keeps ``python single_test.py --help`` fast.
    from lod2_texture_pipeline.pipeline import process_building
    from lod2_texture_pipeline.projection import load_sam3

    device, processor, sam3_prompt_facade, sam3_prompt_roof = load_sam3(
        prompt_facade=SAM3_PROMPT_FACADE,
        prompt_roof=SAM3_PROMPT_ROOF,
    )

    print(
        f"SAM3 loaded on device: {device} | "
        f"facade_prompt={sam3_prompt_facade!r} | "
        f"facade_refinement_prompt={SAM3_PROMPT_FACADE_REFINEMENT!r} | "
        f"roof_prompt={sam3_prompt_roof!r}"
    )

    process_building(
        geojson_path=str(geojson_path),
        out_root=str(args.output_dir),
        geotiff_path=str(geotiff_path) if geotiff_path is not None else None,
        device=device,
        processor=processor,
        sam3_prompt_facade=sam3_prompt_facade,
        sam3_prompt_facade_refinement=SAM3_PROMPT_FACADE_REFINEMENT,
        sam3_prompt_roof=sam3_prompt_roof,
    )


if __name__ == "__main__":
    main()
