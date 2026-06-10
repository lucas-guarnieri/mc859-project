#!/bin/bash
# Full experiment pipeline — run from the project root.
# Each step depends on the previous one completing successfully.
set -e

echo "=== [1/5] Preprocessing ==="
python -m scripts.run_preprocessing

echo "=== [2/5] Building graphs ==="
python -m scripts.build_all_graphs

echo "=== [3/5] Fitting models ==="
python -m scripts.fit_models

echo "=== [4/5] Running experiment ==="
python -m src.experiments.run_experiment

echo "=== [5/5] Generating plots ==="
python -m scripts.generate_report_plots

echo "=== Pipeline complete. Results in results/ ==="
