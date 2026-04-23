# mc859-project

MC859 project analyzing user–item interaction graphs from the Amazon Reviews dataset. Focuses on structural properties, temporal evolution, and path centralization in graph-based recommender systems.

## Overview

This project investigates how user–item interaction graphs evolve over time and how highly connected nodes may concentrate recommendation paths in graph-based recommender systems. The study uses temporal snapshots built from the Amazon Reviews dataset.

Main goals include:

- Build yearly cumulative graph snapshots
- Analyze structural graph properties over time
- Measure degree concentration and connectivity patterns
- Study path centralization effects in recommendation scenarios
- Support future experiments with graph-based recommendation models

## Dataset

Source dataset:

- Amazon Reviews Dataset

Interactions are preprocessed and organized into temporal snapshots, where each graph contains all interactions observed up to a given year.

## Graph Snapshots

Public graph instances are available in the repository Releases section.

Current release:

- **Graph Snapshots v1**
  - `graph_2016.graphml`
  - `graph_2017.graphml`
  - `graph_2018.graphml`

Each file is a cumulative snapshot up to the corresponding year.

## Format

Graphs are stored in **GraphML** format and can be used with:

- Gephi
- NetworkX
- igraph
- Cytoscape

## Repository Structure

```text
data/        raw, processed data and snapshots
src/         preprocessing, graph construction, models, evaluation
notebooks/   exploratory analysis and experiments
results/     plots, metrics, logs
reports/     project reports and figures
configs/     configuration files
```

## Current Analysis

Initial graph analysis includes:

- Number of nodes and edges
- Average degree
- Degree distribution
- Connected components / strongly connected components
- Component size distribution

## Future Work

- Implement recommendation baselines
- Multi-hop graph recommenders
- Penalization of hub nodes
- Precision@K and NDCG@K evaluation
- Diversity and centralization metrics

## Author

Lucas de Lima Guarnieri
