# Licenses: code and data

## Code and precomputed results

The code in this repository and the summary files in `precomputed_results/` are released under the MIT License (see `LICENSE`). The summary files contain aggregate statistics only: counts, shares, intervals and certification outcomes. They contain no raw Criteo rows and no per-journey records.

## Criteo Attribution Modeling for Bidding dataset

The real-data experiments use this public dataset (Diemert, Meynet, Galland and Lefortier, ADKDD 2017). Criteo distributes it under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license (CC-BY-NC-SA-4.0): <https://huggingface.co/datasets/criteo/criteo-attribution-dataset>. It is not redistributed here.

- `scripts/download_criteo.py` downloads it from the original source, pinned to dataset revision `904188a63cbad78bee43cd26ff5ee4ac77903986`, and verifies its size (653,015,824 bytes) and SHA-256 (`94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98`).
- Per-journey files derived from it, such as `results/credits*/**.parquet` and the journey caches, are produced locally by `scripts/materialize_criteo_credits.py` and are not part of this repository. Anyone who regenerates them is bound by the dataset license: non-commercial use, attribution and share-alike.

Commercial use of this code together with the Criteo data is not permitted by the dataset license.
