import networkx as nx
from collections import defaultdict
from src.graph.graph_utils import get_product_nodes, get_user_products


class CooccurrenceRecommender:
    """Item co-occurrence recommender using Jaccard similarity.

    Two products co-occur when at least one user has interacted with both.
    Recommendation score for a candidate product is the sum of Jaccard
    similarities between the candidate and each product in the user's history.
    """

    name = "cooccurrence"
    tracks_paths = False

    def __init__(self):
        self.cooc: dict = {}
        self.item_users: dict = {}
        self._cooc_index: dict = {}

    def fit(self, G: nx.Graph):
        product_nodes = set(get_product_nodes(G))
        self.item_users = {p: set(G.neighbors(p)) for p in product_nodes}

        cooc_counts = defaultdict(float)
        for p, users in self.item_users.items():
            for user in users:
                for other_p in G.neighbors(user):
                    if other_p != p and other_p in product_nodes:
                        cooc_counts[(p, other_p)] += 1

        self.cooc = {}
        for (p1, p2), count in cooc_counts.items():
            union = len(self.item_users[p1]) + len(self.item_users[p2]) - count
            self.cooc[(p1, p2)] = count / union if union > 0 else 0.0

        self._cooc_index = defaultdict(list)
        for (p1, p2), score in self.cooc.items():
            self._cooc_index[p1].append((p2, score))

        return self

    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
        scores = defaultdict(float)
        for item in seen:
            for other_p, s in self._cooc_index.get(item, []):
                if other_p not in seen:
                    scores[other_p] += s
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:k]]
