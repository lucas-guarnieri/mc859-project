import math
import numpy as np
import networkx as nx
from collections import Counter
from typing import Dict
 
 
def path_node_entropy(node_counts: Dict) -> float:
    """
    Shannon entropy of the intermediate node usage distribution.
 
    H = -sum(p_i * log2(p_i))
 
    High entropy -> uniform usage across many nodes (diverse paths)
    Low entropy  -> few nodes dominate all paths (concentrated)
    """
    total = sum(node_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in node_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy
 
 
def path_node_gini(node_counts: Dict) -> float:
    """
    Gini coefficient of the intermediate node usage distribution.
 
    Gini = 0 -> all nodes used equally
    Gini = 1 -> single node used for all paths
    """
    values = sorted(node_counts.values())
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    arr = np.array(values, dtype=float)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)
 
 
def hub_concentration(node_counts: Dict, top_fraction: float = 0.1) -> float:
    """
    Fraction of total path usage accounted for by the top X% most-used nodes.
 
    Example: top_fraction=0.1 -> what fraction of all path uses go through
    the top 10% most-used intermediate nodes?
 
    High value -> recommendations are structurally concentrated on few hubs.
    """
    counts = sorted(node_counts.values(), reverse=True)
    n = len(counts)
    total = sum(counts)
    if n == 0 or total == 0:
        return 0.0
    top_n = max(1, int(n * top_fraction))
    return sum(counts[:top_n]) / total
 
 
def degree_usage_correlation(node_counts: Dict, G: nx.Graph) -> float:
    """
    Pearson correlation between a node's degree and its path usage frequency.
 
    Positive correlation -> high-degree nodes (hubs) dominate paths.
    Near zero            -> usage is independent of degree.
    """
    nodes = list(node_counts.keys())
    if len(nodes) < 2:
        return 0.0
    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    usages = np.array([node_counts[n] for n in nodes], dtype=float)
    if degrees.std() == 0 or usages.std() == 0:
        return 0.0
    return float(np.corrcoef(degrees, usages)[0, 1])
 
 
def centralization_report(
    node_counts: Dict,
    G: nx.Graph,
    top_fraction: float = 0.1
) -> dict:
    """
    Full centralization report for a set of recommendation paths.
    Aggregates all metrics into a single dict.
    """
    return {
        "entropy":           path_node_entropy(node_counts),
        "gini":              path_node_gini(node_counts),
        "hub_concentration": hub_concentration(node_counts, top_fraction),
        "degree_usage_corr": degree_usage_correlation(node_counts, G),
        "n_unique_nodes":    len(node_counts),
        "total_path_uses":   sum(node_counts.values()),
    }
 
 
def aggregate_node_counts(all_node_counts: list) -> Counter:
    """
    Merge a list of per-user node_counts Counters into a single Counter.
    Used to compute centralization metrics across all evaluated users.
    """
    combined = Counter()
    for nc in all_node_counts:
        combined.update(nc)
    return combined
 