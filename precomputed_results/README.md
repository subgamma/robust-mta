# precomputed_results

Summary files behind every figure, table and number of the paper: aggregate statistics only (counts, shares,
intervals, certification outcomes). No raw Criteo rows and no per-journey records. The directory mirrors the
layout of `results/`; `make figures-from-precomputed` copies it into `results/` (without overwriting existing files)
and regenerates the figures, tables and manuscript numbers. The scripts never modify this directory.

- `e1/` ... `e10/`: final outputs of experiments E1 to E10 (`e1_simplified_boundary.json` and `e3_simplified_boundary.json`
  are the looser-boundary comparison runs written with `--boundary simplified`).
- `config_runs/`: the A/B/C/D ablation summary (`comparison.json`) and the configuration-A E5 run used for the reference curve.
- `diagnostics/`: boundary widths, drift diagnostic and drift series, the clipped-legacy check and the Criteo selection check.
- `manuscript_numbers.json`: every number quoted in the paper, with its source file and field.
