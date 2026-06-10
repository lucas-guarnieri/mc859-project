# mc859-project

Graph-based recommender systems on the Amazon Electronics dataset (MC859 — UNICAMP).

The project evaluates multi-hop path traversal models on yearly cumulative bipartite graphs (users ↔ products), measuring how degree-based penalization affects precision, diversity, and path centralization.

## Setup

**Python 3.10+** required.

```bash
pip install -r requirements.txt
```

### Dataset

Download the Amazon Reviews 2018 — Electronics category from:
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Save the raw file as:
```
data/raw/Electronics.csv
```

The expected CSV columns (no header) are: `item_id, user_id, rating, timestamp`.

## Running the Experiment

From the project root, run the full pipeline in sequence:

```bash
bash run_pipeline.sh
```

Or execute each step individually:

```bash
# 1. Filter interactions, encode IDs, and generate yearly snapshots
python -m scripts.run_preprocessing

# 2. Build bipartite graph for each snapshot year
python -m scripts.build_all_graphs

# 3. Fit and cache all models (slow — run once)
python -m scripts.fit_models

# 4. Run temporal evaluation (train year T → test year T+1)
python -m src.experiments.run_experiment

# 5. Generate comparison plots from results
python -m scripts.generate_report_plots
```

All scripts must be run from the **project root** so that `src/` and `configs/` are on the Python path.

### Configuration

All parameters are in `configs/base.yaml`:

| Key | Description |
|---|---|
| `data.raw_path` | Path to the raw CSV file |
| `filtering.min_year` | Earliest year to include |
| `filtering.min_user_interactions` | Minimum interactions per user |
| `filtering.min_item_interactions` | Minimum interactions per item |
| `graph.years` | Years for which to build graphs |
| `experiment.train_years` | Training years for evaluation |
| `experiment.k` | Top-K for recommendations |
| `experiment.n_users` | Users sampled per evaluation period |

## Repository Structure

```
configs/        experiment configuration (base.yaml)
data/
  raw/          raw dataset files (not versioned)
  interim/      encoded ID mappings
  processed/    filtered and encoded interaction table
  snapshots/    yearly cumulative interaction snapshots
  graphs/       GraphML bipartite graph files
results/
  models/       fitted model cache (pickle)
  metrics/      evaluation results (JSON)
  plots/        generated figures
scripts/        top-level executable scripts
src/
  data/         preprocessing, encoding, temporal split
  graph/        graph construction and node utilities
  models/       recommendation models
  evaluation/   metrics, diversity, centralization
  experiments/  evaluation loop and configuration
reports/        project reports and bibliography
run_pipeline.sh full end-to-end pipeline script
```

## Models

| Model | Category |
|---|---|
| Popularity | Baseline |
| Co-occurrence (Jaccard) | Baseline |
| Item-CF (cosine similarity) | Baseline |
| Personalized PageRank | Graph reference |
| Multi-hop (3-hop) | Proposed |
| Multi-hop penalized — sqrt | Proposed |
| Multi-hop penalized — log | Proposed |
| Multi-hop penalized — quadratic | Proposed |

## Evaluation

Temporal evaluation: model trained on year T, evaluated on new interactions from year T+1.
Three periods: 2015→2016, 2016→2017, 2017→2018.

Metrics: Precision@K, Recall@K, NDCG@K, Diversity, Coverage, Recommendation Gini.
Path-based models also report: hub concentration, path node Gini, degree-usage correlation.

Two relevance modes:
- `all` — any new interaction in the test period is relevant
- `rated` — only new interactions with rating ≥ 4.0 are relevant

## Author

Lucas de Lima Guarnieri — l119756@dac.unicamp.br
