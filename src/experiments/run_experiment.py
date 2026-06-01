"""
experiments/run_experiment.py
 
Entry point for the full temporal evaluation pipeline.
 
Usage:
    python -m src.experiments.run_experiment
    python -m src.experiments.run_experiment --config configs/base.yaml
"""
 
import argparse
import logging
import json
import os
 
from src.experiments.config     import load_config, build_models
from src.experiments.temporal_eval import run_temporal_evaluation
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal recommendation experiment")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config file (default: configs/base.yaml)"
    )
    return parser.parse_args()
 
 
def print_summary(all_results: dict):
    """Print a formatted summary table of results."""
    for year, year_results in all_results.items():
        for mode, mode_results in year_results.items():
            print(f"\nTrain {year} -> Test {int(year) + 1} | Mode: {mode}")
            print(f"{'Model':<40} {'Precision':>10} {'NDCG':>10} {'Diversity':>10} {'Rec Gini':>10}")
            print("-" * 85)
            for model_name, metrics in mode_results.items():
                print(
                    f"{model_name:<40}"
                    f" {metrics.get('precision', 0):>10.4f}"
                    f" {metrics.get('ndcg', 0):>10.4f}"
                    f" {metrics.get('diversity_score', 0):>10.4f}"
                    f" {metrics.get('rec_gini', 0):>10.4f}"
                )
 
 
def main():
    args = parse_args()
 
    config = load_config(args.config)
    cfg_exp = config["experiment"]
 
    models = build_models(config)
    logging.info(f"Models to evaluate: {list(models.keys())}")
 
    results = run_temporal_evaluation(
        snapshot_dir = config["data"]["graphs_dir"],
        results_dir  = config["results"]["metrics_dir"],
        models_dir   = config["results"]["models_dir"],
        models       = models,
        k            = cfg_exp["k"],
        n_users      = cfg_exp["n_users"],
        random_seed  = cfg_exp["random_seed"],
        eval_modes   = cfg_exp["eval_modes"],
        years        = cfg_exp["train_years"],
    )
 
    print_summary(results)
 
 
if __name__ == "__main__":
    main()