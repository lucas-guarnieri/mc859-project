import numpy as np
from typing import List, Dict, Set
from collections import Counter
 
 
def intra_list_diversity(
    recommended: List,
    item_users: Dict[str, Set]
) -> float:
    """
    Average pairwise Jaccard distance between recommended items,
    measured over their user sets.
 
    distance(A, B) = 1 - Jaccard(A, B) = 1 - |A ∩ B| / |A ∪ B|
 
    ILD = 0 -> all items share the same users (no diversity)
    ILD = 1 -> no item shares any user (maximum diversity)
    """
    n = len(recommended)
    if n < 2:
        return 0.0
 
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = item_users.get(recommended[i], set())
            b = item_users.get(recommended[j], set())
            union = len(a | b)
            inter = len(a & b)
            jaccard = inter / union if union > 0 else 0.0
            total += 1.0 - jaccard
            count += 1
 
    return total / count if count > 0 else 0.0
 
 
def novelty(
    recommended: List,
    item_popularity: Dict,
    n_users: int
) -> float:
    """
    Average self-information of recommended items.
 
    novelty(item) = -log2(popularity(item) / n_users)
 
    Higher novelty -> less popular items recommended.
    Lower novelty  -> model recommends mostly well-known items.
    """
    if not recommended:
        return 0.0
    scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1) / max(n_users, 1)
        scores.append(-np.log2(max(pop, 1e-10)))
    return float(np.mean(scores))
 
 
def coverage(all_recommended: List[List], catalog_size: int) -> float:
    """
    Fraction of catalog items that appear in at least one recommendation list.
    Higher coverage -> model explores a larger portion of the catalog.
    """
    covered = set(item for recs in all_recommended for item in recs)
    return len(covered) / catalog_size if catalog_size > 0 else 0.0
 
 
def gini_coefficient(values: List[float]) -> float:
    """
    Gini coefficient of a distribution.
    0 -> perfectly equal distribution
    1 -> maximally concentrated (one item gets everything)
    """
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)
 
 
def recommendation_frequency_gini(
    all_recommended: List[List],
    catalog: List
) -> float:
    """
    Gini coefficient over the distribution of how often each catalog item
    is recommended across all users.
 
    High Gini -> few items dominate all recommendation lists.
    Low Gini  -> recommendations are spread across the catalog.
    """
    counter = Counter(item for recs in all_recommended for item in recs)
    freqs = [counter.get(item, 0) for item in catalog]
    return gini_coefficient(freqs)
 
 
def diversity_score(
    recommended: List,
    item_users: Dict[str, Set],
    item_popularity: Dict,
    n_users: int,
    alpha: float = 0.5
) -> float:
    """
    Composite diversity score combining ILD and novelty.
 
    diversity_score = alpha * ILD + (1 - alpha) * novelty_normalized
 
    Both components are normalized to [0, 1].
    alpha = 0.5 by default (equal weight to both components).
    """
    ild = intra_list_diversity(recommended, item_users)
    nov = novelty(recommended, item_popularity, n_users)
 
    # Normalize novelty by its theoretical maximum: log2(n_users)
    max_novelty = np.log2(max(n_users, 2))
    nov_norm = min(nov / max_novelty, 1.0)
 
    return alpha * ild + (1 - alpha) * nov_norm