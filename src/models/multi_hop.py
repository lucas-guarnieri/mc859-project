import networkx as nx
from collections import defaultdict, Counter
from src.graph.graph_utils import get_product_nodes, get_user_products


class MultiHopRecommender:
    """Multi-hop path recommender without hub penalization.

    Traverses 3-hop paths (user→product→user→product) and optionally
    5-hop paths, accumulating inverse-degree weights along each path.

    mode "3"   — 3-hop paths only
    mode "5"   — 5-hop paths only
    mode "sum" — sum of both
    """

    tracks_paths = True

    def __init__(self, mode: str = "3"):
        assert mode in ("3", "5", "sum"), "mode must be '3', '5', or 'sum'"
        self.mode = mode
        self.name = f"multi_hop_{mode}"

    def fit(self, G: nx.Graph):
        return self

    def recommend(self, G: nx.Graph, user, k: int = 10) -> tuple:
        """Return (top-k products, node_counts) for centralization analysis."""
        seen = set(get_user_products(G, user))
        product_nodes = set(get_product_nodes(G))
        deg = dict(G.degree())
        node_counts = Counter()

        scores_3 = defaultdict(float)
        scores_5 = defaultdict(float)

        for p1 in G.neighbors(user):
            if p1 not in product_nodes:
                continue
            w1 = 1.0 / deg[p1]

            for u2 in G.neighbors(p1):
                if u2 == user:
                    continue
                w2 = w1 / deg[u2]

                for p2 in G.neighbors(u2):
                    if p2 not in product_nodes or p2 in seen:
                        continue
                    scores_3[p2] += w2
                    node_counts[p1] += 1
                    node_counts[u2] += 1

                    if self.mode in ("5", "sum"):
                        w3 = w2 / deg[p2]
                        for u3 in G.neighbors(p2):
                            if u3 == user or u3 == u2:
                                continue
                            w4 = w3 / deg[u3]
                            for p3 in G.neighbors(u3):
                                if p3 not in product_nodes or p3 in seen:
                                    continue
                                scores_5[p3] += w4
                                node_counts[p2] += 1
                                node_counts[u3] += 1

        scores = self._aggregate(scores_3, scores_5)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:k]], node_counts

    def _aggregate(self, scores_3: defaultdict, scores_5: defaultdict) -> dict:
        if self.mode == "3":
            return scores_3
        if self.mode == "5":
            return scores_5
        combined = defaultdict(float)
        for p, s in scores_3.items():
            combined[p] += s
        for p, s in scores_5.items():
            combined[p] += s
        return combined
