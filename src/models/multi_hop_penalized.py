import math
import networkx as nx
from collections import defaultdict, Counter
from src.graph.graph_utils import get_product_nodes, get_user_products


def _pen_sqrt(d):      return 1.0 / math.sqrt(max(d, 1))
def _pen_log(d):       return 1.0 / math.log1p(max(d, 1))
def _pen_quadratic(d): return 1.0 / max(d, 1) ** 2

PENALTY_FUNCTIONS = {
    "sqrt":      _pen_sqrt,
    "log":       _pen_log,
    "quadratic": _pen_quadratic,
}


class MultiHopPenalizedRecommender:
    """Multi-hop path recommender with degree-based hub penalization.

    At each node along a path the edge weight is multiplied by a penalty
    function of that node's degree, down-weighting high-degree hubs.

    penalty "sqrt"      — weight *= 1/sqrt(degree)
    penalty "log"       — weight *= 1/log(1 + degree)
    penalty "quadratic" — weight *= 1/degree²
    """

    tracks_paths = True

    def __init__(self, mode: str = "3", penalty: str = "log"):
        assert mode in ("3", "5", "sum"), "mode must be '3', '5', or 'sum'"
        assert penalty in PENALTY_FUNCTIONS, (
            f"penalty must be one of {list(PENALTY_FUNCTIONS.keys())}"
        )
        self.mode = mode
        self.penalty = penalty
        self.name = f"multi_hop_penalized_{penalty}_{mode}"
        self._pen = PENALTY_FUNCTIONS[penalty]

    def fit(self, G: nx.Graph):
        return self

    def recommend(self, G: nx.Graph, user, k: int = 10) -> tuple:
        """Return (top-k products, node_counts) for centralization analysis."""
        seen = set(get_user_products(G, user))
        product_nodes = set(get_product_nodes(G))
        deg = dict(G.degree())
        pen = self._pen
        node_counts = Counter()

        scores_3 = defaultdict(float)
        scores_5 = defaultdict(float)

        for p1 in G.neighbors(user):
            if p1 not in product_nodes:
                continue
            w1 = pen(deg[p1])

            for u2 in G.neighbors(p1):
                if u2 == user:
                    continue
                w2 = w1 * pen(deg[u2])

                for p2 in G.neighbors(u2):
                    if p2 not in product_nodes or p2 in seen:
                        continue
                    scores_3[p2] += w2
                    node_counts[p1] += 1
                    node_counts[u2] += 1

                    if self.mode in ("5", "sum"):
                        w3 = w2 * pen(deg[p2])
                        for u3 in G.neighbors(p2):
                            if u3 == user or u3 == u2:
                                continue
                            w4 = w3 * pen(deg[u3])
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
