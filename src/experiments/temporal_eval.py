import gc
import os
import json
import pickle
import random
import logging
import networkx as nx

from typing import Dict, List

from src.graph.graph_utils import (
    get_user_nodes, get_product_nodes, get_user_products
)
from src.evaluation.metrics       import evaluate_user, aggregate_metrics
from src.evaluation.diversity     import (
    diversity_score, recommendation_frequency_gini, coverage
)
from src.evaluation.centralization import (
    centralization_report, aggregate_node_counts
)

logger = logging.getLogger(__name__)

RATING_THRESHOLD = 4.0


def build_test_set(
    train_G: nx.Graph,
    test_G: nx.Graph,
    mode: str = "all"
) -> Dict[str, set]:
    """Extract new interactions in test_G not present in train_G.

    mode "all"   — every new interaction is relevant
    mode "rated" — only interactions with rating >= 4.0 are relevant

    Returns {user: set_of_relevant_products} for users present in train_G.
    """
    assert mode in ("all", "rated")

    train_users = set(get_user_nodes(train_G))
    train_edges = set(train_G.edges())

    test_set = {}
    for u, v, data in test_G.edges(data=True):
        user    = u if u in train_users else v
        product = v if u in train_users else u

        if user not in train_users:
            continue
        if (u, v) in train_edges or (v, u) in train_edges:
            continue
        if mode == "rated" and float(data.get("rating", 0.0)) < RATING_THRESHOLD:
            continue

        test_set.setdefault(user, set()).add(product)

    return test_set


def sample_users(test_set: Dict[str, set], n: int, seed: int) -> Dict[str, set]:
    """Sample up to n users from the test set with a fixed random seed."""
    random.seed(seed)
    users = list(test_set.keys())
    sampled = random.sample(users, min(n, len(users)))
    return {u: test_set[u] for u in sampled}


def load_or_fit(model, train_G: nx.Graph, models_dir: str, year: int):
    """Load a fitted model from cache, or fit and save it if not cached."""
    pkl_path = os.path.join(models_dir, f"{model.name}_{year}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            fitted = pickle.load(f)
        logger.info(f"    Loaded {model.name} from cache.")
        return fitted
    model.fit(train_G)
    os.makedirs(models_dir, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"    Saved {model.name} to cache.")
    return model


def evaluate_model(
    model,
    train_G: nx.Graph,
    test_set: Dict[str, set],
    product_nodes: set,
    k: int,
) -> dict:
    """Generate recommendations and compute all metrics for a fitted model."""
    item_users = {p: set(train_G.neighbors(p)) for p in product_nodes}
    item_popularity = {p: len(item_users[p]) for p in product_nodes}
    n_users = len(get_user_nodes(train_G))
    catalog = list(product_nodes)

    if hasattr(model, 'precompute'):
        model.precompute(list(test_set.keys()))

    user_results = []
    all_recs = []
    all_node_counts = []

    for user, relevant in test_set.items():
        if model.tracks_paths:
            recs, node_counts = model.recommend(train_G, user, k=k)
            all_node_counts.append(node_counts)
        else:
            recs = model.recommend(train_G, user, k=k)

        all_recs.append(recs)
        result = evaluate_user(recs, relevant, k=k)
        result["diversity_score"] = diversity_score(recs, item_users, item_popularity, n_users)
        user_results.append(result)

    agg = aggregate_metrics(user_results)
    agg["coverage"] = coverage(all_recs, len(catalog))
    agg["rec_gini"] = recommendation_frequency_gini(all_recs, catalog)

    if model.tracks_paths and all_node_counts:
        combined_counts = aggregate_node_counts(all_node_counts)
        agg["centralization"] = centralization_report(combined_counts, train_G)

    return agg


def run_temporal_evaluation(
    snapshot_dir: str,
    results_dir: str,
    models_dir: str,
    models: dict,
    k: int = 10,
    n_users: int = 5000,
    random_seed: int = 42,
    eval_modes: List[str] = ["all", "rated"],
    years: List[int] = [2015, 2016, 2017],
) -> dict:
    """Run the full temporal evaluation loop.

    For each train year t, evaluates all models on interactions from year t+1.
    Saves one JSON file per (year, mode) combination to results_dir.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    total_steps = len(years) * len(eval_modes) * len(models)
    completed_steps = 0
    last_notified_pct = -1

    def _notify(label: str):
        nonlocal last_notified_pct
        pct = int(completed_steps / total_steps * 100)
        threshold = (pct // 20) * 20
        if threshold > last_notified_pct and threshold > 0:
            last_notified_pct = threshold
            logger.info(f"PROGRESS {threshold}% | {label}")

    for year in years:
        train_path = os.path.join(snapshot_dir, f"graph_{year}.graphml")
        test_path  = os.path.join(snapshot_dir, f"graph_{year + 1}.graphml")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.warning(f"Missing graph files for {year}/{year + 1}, skipping.")
            continue

        logger.info(f"=== Train: {year} | Test: {year + 1} ===")
        train_G = nx.read_graphml(train_path)
        test_G  = nx.read_graphml(test_path)
        product_nodes = set(get_product_nodes(train_G))

        logger.info(f"  Fitting models for year {year}...")
        fitted_models = {
            name: load_or_fit(model, train_G, models_dir, year)
            for name, model in models.items()
        }

        year_results = {}
        for mode in eval_modes:
            logger.info(f"  Evaluation mode: {mode}")
            test_set = build_test_set(train_G, test_G, mode=mode)
            test_set = sample_users(test_set, n=n_users, seed=random_seed)
            logger.info(f"  Sampled users: {len(test_set)}")

            if not test_set:
                logger.warning(f"  No test users for mode={mode}, skipping.")
                continue

            mode_results = {}
            for model_name, model in fitted_models.items():
                logger.info(f"    Evaluating {model_name}...")
                agg = evaluate_model(model, train_G, test_set, product_nodes, k=k)
                agg["mode"] = mode
                mode_results[model_name] = agg
                logger.info(
                    f"    {model_name}: "
                    f"precision={agg.get('precision', 0):.4f} | "
                    f"ndcg={agg.get('ndcg', 0):.4f} | "
                    f"diversity={agg.get('diversity_score', 0):.4f}"
                )
                if hasattr(model, '_ppr_cache'):
                    model._ppr_cache = {}
                completed_steps += 1
                _notify(f"{year}->{year+1} | mode={mode} | last={model_name}")

            year_results[mode] = mode_results

            out_path = os.path.join(results_dir, f"results_{year}_{year + 1}_{mode}.json")
            with open(out_path, "w") as f:
                json.dump(mode_results, f, indent=2)
            logger.info(f"  Saved: {out_path}")

        all_results[str(year)] = year_results

        del train_G, test_G, fitted_models, product_nodes
        gc.collect()

    combined_path = os.path.join(results_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {combined_path}")

    return all_results
