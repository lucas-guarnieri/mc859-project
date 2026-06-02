"""
models/cooccurrence.py

Item co-occurrence recommender.
Two products co-occur when at least one user has interacted with both.
Similarity score: Jaccard coefficient between the user sets of each product pair.
"""

import networkx as nx
from collections import defaultdict
from src.graph.graph_utils import get_product_nodes, get_user_products

class CooccurrenceRecommender:

    name = "cooccurrence"
    tracks_paths = False

    def __init__(self):
        self.cooc: dict = {}        # (product1, product2) -> jaccard score
        self.item_users: dict = {}  # product -> set of users
        self._cooc_index: dict = {} # product -> [(other_product, score)] for fast lookup

    def fit(self, G: nx.Graph):
        product_nodes = set(get_product_nodes(G))
        self.item_users = {
            p: set(G.neighbors(p)) for p in product_nodes
        }

        cooc_counts = defaultdict(float)
        for p, users in self.item_users.items():
            for user in users:
                for other_p in G.neighbors(user):
                    if other_p != p and other_p in product_nodes:
                        cooc_counts[(p, other_p)] += 1

        # Compute Jaccard similarity for each co-occurring pair
        # Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        # where A and B are the user sets of each product
        self.cooc = {}
        for (p1, p2), count in cooc_counts.items():
            union = len(self.item_users[p1]) + len(self.item_users[p2]) - count
            self.cooc[(p1, p2)] = count / union if union > 0 else 0.0

        # Build forward index: product -> list of (candidate, score)
        # Avoids iterating the full catalog in recommend()
        self._cooc_index = defaultdict(list)
        for (p1, p2), score in self.cooc.items():
            self._cooc_index[p1].append((p2, score))

        return self

    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        if not hasattr(self, '_cooc_index') or not self._cooc_index:
            self._cooc_index = defaultdict(list)
            for (p1, p2), score in self.cooc.items():
                self._cooc_index[p1].append((p2, score))

        seen = set(get_user_products(G, user))
        scores = defaultdict(float)
        for item in seen:
            for other_p, s in self._cooc_index.get(item, []):
                if other_p not in seen:
                    scores[other_p] += s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:k]]
