# -*- coding: utf-8 -*-

from pathlib import Path

from lod2_texture_pipeline.pipeline import process_building
from lod2_texture_pipeline.projection import load_sam3
from lod2_texture_pipeline.config import OUTPUT_DIR, SAM3_PROMPT_FACADE, SAM3_PROMPT_ROOF


REPO_ROOT = Path(__file__).resolve().parent

GEOJSON_PATH = REPO_ROOT / "sample_data" / "3d_geojsons" / "building_48959353_3d.geojson"
GEOTIFF_PATH = REPO_ROOT / "sample_data" / "geotiffs" / "building_48959353.tif"


def main():
    device, processor, sam3_prompt_facade, sam3_prompt_roof = load_sam3(
        prompt_facade=SAM3_PROMPT_FACADE,
        prompt_roof=SAM3_PROMPT_ROOF,
    )

    print(
        f"✅ SAM3 loaded on device: {device} | "
        f"facade_prompt={sam3_prompt_facade!r} | "
        f"roof_prompt={sam3_prompt_roof!r}"
    )

    process_building(
        geojson_path=str(GEOJSON_PATH),
        out_root=OUTPUT_DIR,
        geotiff_path=str(GEOTIFF_PATH),
        device=device,
        processor=processor,
        sam3_prompt_facade=sam3_prompt_facade,
        sam3_prompt_roof=sam3_prompt_roof,
    )


if __name__ == "__main__":
    main()
