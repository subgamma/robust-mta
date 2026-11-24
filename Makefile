# Run from the repository root; see README.md and REPRODUCIBILITY.md for the three reproduction tracks.
PY ?= python
.PHONY: install test proof-checks reproduce-synthetic download-criteo materialize-criteo train-darnn \
        reproduce-criteo ablation diagnostics figures-from-precomputed figures

install:
	pip install -r requirements.txt
	pip install -e .

test:
	$(PY) -m pytest -q

proof-checks:
	$(PY) -I scripts/proof_checks/independent_proof_audit_checks.py

# Track B: synthetic experiments E1-E4 and E10 (no external data), about 8-10 minutes.
reproduce-synthetic:
	$(PY) scripts/run_e1.py --reps 200 --workers 8
	$(PY) -m anytime_attribution.experiments.e2_optional_stopping
	$(PY) scripts/run_e3.py --reps 200
	$(PY) -m anytime_attribution.experiments.e4_confident_reversal
	$(PY) scripts/run_e10_oracle.py
	$(PY) scripts/run_e10_multicampaign.py --n-mon 50000 --reps 100 --workers 12
	$(PY) scripts/run_e10_multicampaign.py --n-mon 250000 --reps 100 --horizons 50000,250000 --workers 12
	$(PY) scripts/e10_decomposition.py 50000
	$(PY) scripts/e10_decomposition.py 250000 results/e10/e10_results_N250000_pair_outcomes.npz

# Track C: Criteo experiments (about 1 hour after the download).
download-criteo:
	$(PY) scripts/download_criteo.py

materialize-criteo:
	$(PY) scripts/materialize_criteo_credits.py

# DARNN credits for the secondary construction (read by E8; seconds per seed on a CPU, GPU/MPS optional).
train-darnn:
	$(PY) scripts/train_darnn_gpu.py

reproduce-criteo: materialize-criteo train-darnn ablation diagnostics

# Final configuration D of E4-E9, then the A/B/C ablation runs on identical streams (the E10 runs need the oracle and the N = 50,000 monitoring run; both are synthetic).
ablation:
	$(PY) scripts/run_e10_oracle.py
	$(PY) scripts/run_e10_multicampaign.py --n-mon 50000 --reps 100 --workers 12
	$(PY) -m anytime_attribution.experiments.e4_criteo_intervention
	for e in e5 e6 e7 e8 e9; do $(PY) scripts/run_config.py D $$e || exit 1; done
	for c in A B C; do \
	  for e in e5 e6 e7; do $(PY) scripts/run_config.py $$c $$e || exit 1; done; \
	  $(PY) scripts/run_e10_multicampaign.py --config $$c || exit 1; \
	done
	$(PY) scripts/run_e10_multicampaign.py --config B --truth pop1m --out results/config_runs/B_pop1m/e10

diagnostics:
	$(PY) scripts/compare_configs.py
	$(PY) scripts/boundary_width.py
	$(PY) scripts/drift_diagnostic.py
	$(PY) scripts/hybrid_region_check.py
	$(PY) scripts/extra_diagnostics.py
	$(PY) scripts/criteo_selection_check.py

# Track A: regenerate every figure, table and manuscript number from the shipped summary files (no Criteo data).
figures-from-precomputed:
	mkdir -p results
	cp -Rn precomputed_results/. results/
	$(MAKE) figures

figures:
	$(PY) scripts/make_tables.py
	$(PY) scripts/make_figures.py
	$(PY) scripts/polish_figures.py
	$(PY) scripts/build_manuscript_numbers.py
	$(PY) scripts/verify_numbers.py
