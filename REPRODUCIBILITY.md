Run all commands from the repository root.

## Agreement levels

- Exact: identical output files, or identical values when the JSON is compared with tolerance 0 (`wall_time*` fields are ignored). This holds on CPU with the versions pinned in `requirements.txt`.
- Float32 tolerance: DARNN credits on Apple MPS or CUDA (see Track C).

Track A re-renders stored results and Track B uses no real data. Only Track C reproduces results from the raw file.

## Track A: stored results to figures, tables and numbers

Needs nothing beyond the install.

```bash
make proof-checks
make figures-from-precomputed
python -m pytest -q
```

`make figures-from-precomputed` copies `precomputed_results/` into `results/` without overwriting, then writes `figures/` (PDF and PNG), `generated_tables/` (`numbers.tex` and `tab_*.tex`) and `results/manuscript_numbers.json`. `precomputed_results/` is never modified. All three output directories are git-ignored.

Expected:

- `cmp results/manuscript_numbers.json precomputed_results/manuscript_numbers.json` prints nothing (161 numbers, 39 summary files).
- `scripts/verify_numbers.py` ends with `items audited: 38; mismatches/unregistered: 0`.
- 31 tests skip without the Criteo file or its credits. With the data present, 326 pass and 2 skip.

Agreement: exact.

## Track B: synthetic experiments

Needs nothing beyond the install.

```bash
make reproduce-synthetic
python scripts/compare_with_precomputed.py
```

| Experiment | Command | Time |
|---|---|---|
| E1 coverage and width | `python scripts/run_e1.py --reps 200 --workers 8` | 80 s |
| E2 optional stopping | `python -m anytime_attribution.experiments.e2_optional_stopping` | 33 s |
| E3 identified sets | `python scripts/run_e3.py --reps 200` | 35 s |
| E4 synthetic | `python -m anytime_attribution.experiments.e4_confident_reversal` | 173 s |
| E10 oracle | `python scripts/run_e10_oracle.py` | 7 s |
| E10 monitoring, N = 50,000 and 250,000, and decomposition | `scripts/run_e10_multicampaign.py`, `scripts/e10_decomposition.py` | 54 s, 107 s |

Outputs go to `results/e1` through `results/e10`. All stochastic steps are seeded from the configuration, and the confidence-sequence constants are fixed in advance. Every output file matches `precomputed_results/` with 0 numeric differences. Files for tracks not run are listed as missing, not as differences.

Agreement: exact.

## Track C: Criteo experiments

Needs the Criteo file (see Data in `README.md`).

```bash
make download-criteo
make reproduce-criteo
make figures
python scripts/compare_with_precomputed.py
```

`make reproduce-criteo` runs `materialize-criteo`, `train-darnn`, `ablation` and `diagnostics`.

| Step | Time |
|---|---|
| E4 Criteo | 578 s |
| E5 | 83 s |
| E6 | 15 s |
| E7 | 184 s |
| E8 | 13 s |
| E9 | 15 s |
| Each ablation configuration A, B, C (E5 to E7 and E10) | about 5 min |

The ablation reruns E5 to E7 and E10 on identical streams under four configurations: A is the ratio-box rule with the simplified boundary, B the hybrid rule with the simplified boundary, C the ratio-box rule with the exact boundary, and D the hybrid rule with the exact boundary (the final configuration). Result files record the label under the key `configuration`.

Offline: download the file from the Hugging Face page, save it as `data/criteo_attribution_dataset.tsv.gz`, and run `python scripts/download_criteo.py --check-only`.

### Exact on CPU

A complete fresh CPU run (credits, five-seed DARNN, E4 to E9, the A/B/C ablation, diagnostics, tables, figures) matches all 39 summary files in `precomputed_results/` with tolerance 0. A repeat of E5 to E9 on the same credits is also identical.

### Hardware-dependent floating point

Last-touch, linear and time-decay credits are deterministic. Logistic+SHAP (seed 0 from `config/criteo.yaml`) and DARNN (`torch.manual_seed(seed)`, full-batch Adam, 5 steps, no shuffling) are seeded and exact on CPU. DARNN output differs between CPU, Apple MPS and CUDA at float32 rounding level: CPU versus MPS differs by 5.1e-8 per journey credit and 1e-10 in aggregate share, so E8 and the DARNN credit files agree across backends only to that tolerance. CUDA has not been tested. E8 reports seed-level statistics.

### Journey ordering

The released Criteo file is sorted by impression timestamp and has no timestamp ties inside a journey. Ties occur across journeys: 3,676 conversion times are shared, covering 7,442 of the 109,221 primary journeys. Polars `group_by` and `unique` do not guarantee output order, so a stable sort on conversion time alone leaves tied journeys in arbitrary order.

The code fixes this with `ORDER_RULE = "conv_time_then_row_ordinal_v1"` in `anytime_attribution/criteo.py`. Each row keeps its ordinal in the released file. Journeys are ordered by (conversion timestamp, smallest row ordinal of the journey), touches by (timestamp, row ordinal), and sampled populations are sorted before sampling with the seeded generator. The result depends only on the file, not on library version, thread count or hash seed, and the credit files are byte-identical across repeated and concurrent runs. It is a convention, not a claim about the true order of events within one second.

The rule matters for results that depend on a sample drawn by position, such as the 5,000-journey held-out sample behind the empirical re-thinning margin `Gamma_hat_sens` for last-touch and logistic+SHAP. A different valid tie order would move those quantities within Monte Carlo variation, which is why the draw is fixed. Linear and time-decay certification enters through prefix sums of per-journey masses and is insensitive to the order inside a tie group.

## Notes

- `scripts/run_e1.py` and `scripts/run_e3.py` accept `--boundary {exact,simplified}` (default `exact`). The simplified boundary is the looser upper bound used only for the width comparison in E1 and E3.
- Guarantees apply to linear and timestamp-fixed time-decay attribution. Last-touch, logistic+SHAP and DARNN are empirical baselines and are labelled so in every output.

## Where each result comes from

| Item | Generated by |
|---|---|
| E1 coverage and width | `scripts/run_e1.py`, `scripts/make_figures.py`, `scripts/polish_figures.py` |
| E4 confident reversal (synthetic and Criteo) | `experiments/e4_confident_reversal.py`, `experiments/e4_criteo_intervention.py` |
| E5 certification (Criteo) | `scripts/run_config.py D e5`, `scripts/polish_figures.py` |
| E7 stress tests | `scripts/run_config.py D e7`, `scripts/extra_diagnostics.py` (G3 severity) |
| E10 decomposition and decision cost | `scripts/run_e10_oracle.py`, `scripts/run_e10_multicampaign.py`, `scripts/e10_decomposition.py`, `scripts/extra_diagnostics.py` |
| E2, E3, E6, E8, E9 | `e2_optional_stopping.py`, `scripts/run_e3.py`, `scripts/run_config.py D e6`, `e8`, `e9` |
| Ablation table, boundary widths, drift, clipped-legacy check, selection check | `scripts/compare_configs.py`, `boundary_width.py`, `drift_diagnostic.py`, `hybrid_region_check.py`, `criteo_selection_check.py` |
| Counterexamples and lemma checks | `scripts/proof_checks/independent_proof_audit_checks.py` |
| Numbers quoted in the text | `scripts/build_manuscript_numbers.py` |

## Layout

```
anytime_attribution/        library: journeys, estimands, identification, confidence sequences, certification, models, Criteo ingestion
anytime_attribution/experiments/   E1 to E10 modules
config/criteo.yaml          frozen Criteo configuration
scripts/                    runners, diagnostics, builders, download and materialization
scripts/proof_checks/       standalone exact checks
tests/                      pytest suite
docs/criteo_preprocessing.md   preprocessing funnel and design constants (read by the number builder)
precomputed_results/        summary JSON and CSV files, about 2.4 MB
```
