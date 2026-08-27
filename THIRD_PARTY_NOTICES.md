# Third-party notices

## LaMa image inpainting

`lod2_texture_pipeline/big_lama.py` contains a modified, inference-only
adaptation of generator and feature-refinement code from:

- [advimman/lama](https://github.com/advimman/lama)
- [geomagical/lama-with-refiner](https://github.com/geomagical/lama-with-refiner)

The upstream projects are licensed under the Apache License 2.0. A copy is
provided in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

Copyright 2021 Samsung Research and the respective upstream contributors.

The adapted file removes the upstream training stack and retains only the
runtime components used by this pipeline. Model weights are downloaded
separately and are not distributed by this repository.

This notice covers the identified third-party portions only. It does not grant
a license for the repository's original code as a whole.
