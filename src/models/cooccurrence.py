"""
models/cooccurrence.py
 
Item co-occurrence recommender.
Two products co-occur when at least one user has interacted with both.
Similarity score: Jaccard coefficient between the user sets of each product pair.
"""
 
import networkx as nx
from collections import defaultdict
from src.graph.graph_utils import get_product_nodes, get_user_products
from src.utils.io import load_config, save_model, load_model

config = load_config("configs/base.yaml")

class CooccurrenceRecommender:

    name = "cooccurrence"
 
    def __init__(self):
        self.cooc: dict = {}        # (product1, product2) -> jaccard score
        self.item_users: dict = {}  # product -> set of users
 
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
 
        return self
 
    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
        scores = defaultdict(float)
        for item in seen:
            for other_p in self.item_users:
                if other_p not in seen:
                    s = self.cooc.get((item, other_p), 0.0)
                    if s > 0:
                        scores[other_p] += s
 
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:k]]