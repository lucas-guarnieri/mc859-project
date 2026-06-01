import math
import networkx as nx
from collections import defaultdict
from src.graph.graph_utils import get_product_nodes, get_user_products
 
 
PENALTY_FUNCTIONS = {
    "sqrt":      lambda d: 1.0 / math.sqrt(max(d, 1)),
    "log":       lambda d: 1.0 / math.log1p(max(d, 1)),
    "quadratic": lambda d: 1.0 / max(d, 1) ** 2,
}
 
 
class MultiHopPenalizedRecommender:
 
    def __init__(self, mode: str = "sum", penalty: str = "log"):
        """
        mode:    "3" | "5" | "sum"
        penalty: "sqrt" | "log" | "quadratic"
        """
        assert mode in ("3", "5", "sum"), "mode must be '3', '5', or 'sum'"
        assert penalty in PENALTY_FUNCTIONS, (
            f"penalty must be one of {list(PENALTY_FUNCTIONS.keys())}"
        )
        self.mode = mode
        self.penalty = penalty
        self.name = f"multi_hop_penalized_{penalty}_{mode}"
        self._pen = PENALTY_FUNCTIONS[penalty]
 
    def fit(self, G: nx.Graph):
        # No global precomputation needed; paths are explored per user
        return self
 
    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
        product_nodes = set(get_product_nodes(G))
        deg = dict(G.degree())
        pen = self._pen
 
        scores_3 = defaultdict(float)
        scores_5 = defaultdict(float)
 
        for p1 in G.neighbors(user):                      # depth 1: seen product
            if p1 not in product_nodes:
                continue
            w1 = pen(deg[p1])
 
            for u2 in G.neighbors(p1):                    # depth 2: similar user
                if u2 == user:
                    continue
                w2 = w1 * pen(deg[u2])
 
                for p2 in G.neighbors(u2):                # depth 3: candidate product
                    if p2 not in product_nodes or p2 in seen:
                        continue
 
                    scores_3[p2] += w2
 
                    if self.mode in ("5", "sum"):
                        w3 = w2 * pen(deg[p2])
 
                        for u3 in G.neighbors(p2):        # depth 4: another user
                            if u3 == user or u3 == u2:
                                continue
                            w4 = w3 * pen(deg[u3])
 
                            for p3 in G.neighbors(u3):    # depth 5: candidate product
                                if p3 not in product_nodes or p3 in seen:
                                    continue
 
                                scores_5[p3] += w4
 
        scores = self._aggregate(scores_3, scores_5)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:k]]
 
    def _aggregate(
        self,
        scores_3: defaultdict,
        scores_5: defaultdict
    ) -> dict:
        if self.mode == "3":
            return scores_3
        if self.mode == "5":
            return scores_5
 
        # mode == "sum": merge both score dicts
        combined = defaultdict(float)
        for p, s in scores_3.items():
            combined[p] += s
        for p, s in scores_5.items():
            combined[p] += s
        return combined