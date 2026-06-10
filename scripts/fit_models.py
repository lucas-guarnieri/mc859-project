import argparse
import logging
import os
import pickle

import networkx as nx

from src.experiments.config import load_config, build_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit and cache all models")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    models_dir  = config["results"]["models_dir"]
    graphs_dir  = config["data"]["graphs_dir"]
    train_years = config["experiment"]["train_years"]

    os.makedirs(models_dir, exist_ok=True)
    models = build_models(config)

    for year in train_years:
        graph_path = os.path.join(graphs_dir, f"graph_{year}.graphml")
        if not os.path.exists(graph_path):
            logger.warning(f"Graph not found: {graph_path}, skipping.")
            continue

        logger.info(f"=== Loading graph {year} ===")
        G = nx.read_graphml(graph_path)

        for name, model in models.items():
            pkl_path = os.path.join(models_dir, f"{name}_{year}.pkl")
            if os.path.exists(pkl_path):
                logger.info(f"  {name}_{year}: already cached, skipping.")
                continue
            logger.info(f"  Fitting {name}_{year}...")
            model.fit(G)
            tmp_path = pkl_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(model, f)
            os.replace(tmp_path, pkl_path)
            logger.info(f"  Saved {pkl_path}")

    logger.info("Done. All models cached.")


if __name__ == "__main__":
    main()
