import numpy as np
from typing import List, Set
 
 
def precision_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Fraction of top-k recommended items that are relevant.
    """
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0
 
 
def recall_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Fraction of relevant items found in the top-k recommendations.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)
 
 
def ndcg_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Binary relevance: 1 if item in relevant, 0 otherwise.
 
    DCG  = sum( 1 / log2(i + 2) ) for each relevant item at position i
    IDCG = DCG of the ideal ranking (all relevant items at the top)
    NDCG = DCG / IDCG
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
 
 
def hit_rate_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    1 if at least one relevant item is in the top-k, else 0.
    """
    return float(bool(set(recommended[:k]) & relevant))
 
 
def evaluate_user(recommended: List, relevant: Set, k: int = 10) -> dict:
    """
    Compute all metrics for a single user.
    Returns a dict with precision, recall, ndcg, and hit_rate.
    """
    return {
        "precision": precision_at_k(recommended, relevant, k),
        "recall":    recall_at_k(recommended, relevant, k),
        "ndcg":      ndcg_at_k(recommended, relevant, k),
        "hit_rate":  hit_rate_at_k(recommended, relevant, k),
    }
 
 
def aggregate_metrics(user_results: List[dict]) -> dict:
    """
    Average metrics across all evaluated users.
    """
    if not user_results:
        return {}
    keys = user_results[0].keys()
    return {
        k: float(np.mean([r[k] for r in user_results]))
        for k in keys
    }