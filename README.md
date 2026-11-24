# Robust Multi-Touch Attribution

This repository implements partial-identification bounds for aggregate multi-touch attribution shares under differential observability, exact empirical-Bernstein confidence sequences (Howard et al., 2021) with a stitched boundary, and a hybrid anytime-valid certification rule with a robustness radius. It contains the synthetic benchmarks (E1 to E4, E10) and the deterministic Criteo experiments (E4 to E9) that exercise them, together with the summary results behind every reported figure, table and number. It is research code, not a production system.

## Installation

```bash
git clone https://github.com/subgamma/robust-mta.git
cd robust-mta
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

`make install` runs the last two commands. Run everything from the repository root.

## Quick Start

```bash
make proof-checks              # exact checks of the identification lemmas and counterexamples
python -m pytest -q            # 298 passed, 31 skipped
make figures-from-precomputed  # tables, figures and numbers from precomputed_results/
```

The 31 skipped tests need the raw Criteo file or the credits derived from it.

## Reproducing Results

| Track | Reproduces | Needs | Agreement |
|---|---|---|---|
| A | Figures, tables and numbers from the stored summaries | nothing | exact |
| B | E1 to E4 (synthetic) and E10 from scratch | nothing | exact |
| C | E4 to E9 on Criteo, ablation, diagnostics | Criteo file (653 MB) | exact on CPU; DARNN agrees to float32 tolerance on MPS or CUDA |

Track A re-renders stored results; only Track C reproduces from raw data. Commands, expected outputs and tolerances are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Data

The Criteo experiments use the [Criteo Attribution Modeling for Bidding dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset) (Diemert et al., ADKDD 2017), released under CC-BY-NC-SA-4.0. Raw data is not redistributed here. `scripts/download_criteo.py` fetches it from Hugging Face at a pinned dataset revision and checks the file size and SHA-256 digest:

```bash
python scripts/download_criteo.py              # writes data/criteo_attribution_dataset.tsv.gz
python scripts/download_criteo.py --check-only # verifies an existing file
```

`precomputed_results/` holds aggregate statistics only, with no raw rows and no per-journey records.

## License

Code and `precomputed_results/` are released under the MIT License ([LICENSE](LICENSE)). The Criteo dataset keeps its own CC-BY-NC-SA-4.0 license, which restricts use of the data and anything derived from it per journey to non-commercial purposes; see [DATA_LICENSE.md](DATA_LICENSE.md).
