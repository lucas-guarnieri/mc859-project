import networkx as nx
from collections import Counter
from src.graph.graph_utils import get_product_nodes, get_user_products


class PopularityRecommender:

    name = "popularity"
    tracks_paths = False

    def __init__(self):
        self.scores: dict = {}

    def fit(self, G: nx.Graph):
        product_nodes = set(get_product_nodes(G))
        counts = Counter()

        for node in product_nodes:
            counts[node] = G.degree(node)

        self.scores = dict(counts)
        return self

    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
        candidates = [
            (p, s) for p, s in self.scores.items() if p not in seen
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in candidates[:k]]