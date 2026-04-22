# -*- coding: utf-8 -*-

from lod2_texture_pipeline.pipeline import process_building
from lod2_texture_pipeline.config import GEOJSON_DIR, GEOTIFF_DIR, OUTPUT_DIR


def main():
    process_building(
        geojson_path=GEOJSON_DIR,
        geotiff_path=GEOTIFF_DIR,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()