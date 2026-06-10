import networkx as nx
from collections import Counter
from src.graph.graph_utils import get_product_nodes, get_user_products


class PopularityRecommender:
    """Recommends the globally most interacted-with products.

    Score: degree of each product node in the training graph.
    Already-seen products are excluded from each user's list.
    """

    name = "popularity"
    tracks_paths = False

    def __init__(self):
        self.scores: dict = {}

    def fit(self, G: nx.Graph):
        product_nodes = set(get_product_nodes(G))
        self.scores = {node: G.degree(node) for node in product_nodes}
        return self

    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
        candidates = [(p, s) for p, s in self.scores.items() if p not in seen]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in candidates[:k]]
