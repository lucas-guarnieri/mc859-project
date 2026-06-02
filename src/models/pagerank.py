import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from collections import defaultdict
from src.graph.graph_utils import get_product_nodes, get_user_products


class PersonalizedPageRankRecommender:

    name = "pagerank"
    tracks_paths = False

    def __init__(self, alpha: float = 0.85, top_k_ppr: int = 500):
        self.alpha = alpha
        self.top_k_ppr = top_k_ppr
        self.pyg_data = None
        self.node_list: list = []
        self.node_index: dict = {}
        self.product_set: set = set()
        self._ppr_cache: dict = {}
        self._edge_index = None

    def fit(self, G: nx.Graph):
        self.node_list = list(G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.node_list)}
        self.product_set = set(get_product_nodes(G))
        self.pyg_data = from_networkx(G)
        self.pyg_data.num_nodes = G.number_of_nodes()
        self._ppr_cache = {}
        return self

    def precompute(self, users: list):
        """Compute PPR for all users at once using sparse matrix multiplication."""
        num_nodes = len(self.node_list)
        # Use saved edge_index if pyg_data was already released
        if self.pyg_data is not None:
            edge_index = self.pyg_data.edge_index.clone()
            self._edge_index = edge_index
            self.pyg_data = None
        else:
            edge_index = self._edge_index
        row, col = edge_index

        # Build sparse row-normalized adjacency matrix A_norm
        # A_norm[dest, src] = 1/deg[src]  for each edge src->dest
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.float32))
        deg_inv = 1.0 / deg.clamp(min=1.0)
        A_norm = torch.sparse_coo_tensor(
            torch.stack([col, row]),
            deg_inv[row],
            (num_nodes, num_nodes),
        ).coalesce()

        # Process users in chunks to keep memory bounded (~200MB per chunk)
        chunk_size = 100
        self._ppr_cache = {}
        for start in range(0, len(users), chunk_size):
            chunk = users[start:start + chunk_size]
            n = len(chunk)
            user_indices = torch.tensor([self.node_index[u] for u in chunk], dtype=torch.long)
            P = torch.zeros(num_nodes, n, dtype=torch.float32)
            P[user_indices, torch.arange(n)] = 1.0

            scores = P.clone()
            for _ in range(100):
                new_scores = self.alpha * torch.sparse.mm(A_norm, scores) + (1 - self.alpha) * P
                if torch.norm(new_scores - scores, p=1) < 1e-6:
                    break
                scores = new_scores

            for i, u in enumerate(chunk):
                self._ppr_cache[u] = scores[:, i]

    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))

        if user in self._ppr_cache:
            ppr_scores = self._ppr_cache[user]
        else:
            num_nodes = len(self.node_list)
            personalization = torch.zeros(num_nodes, dtype=torch.float32)
            personalization[self.node_index[user]] = 1.0
            edge_index = self._edge_index if self.pyg_data is None else self.pyg_data.edge_index
            ppr_scores = self._compute_ppr(edge_index, personalization, num_nodes)

        candidates = [(p, float(ppr_scores[self.node_index[p]]))
                      for p in self.product_set if p not in seen]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in candidates[:k]]

    def _compute_ppr(
        self,
        edge_index: torch.Tensor,
        personalization: torch.Tensor,
        num_nodes: int,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> torch.Tensor:
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.float32))
        deg_inv = 1.0 / deg.clamp(min=1.0)

        scores = personalization.clone()
        for _ in range(max_iter):
            messages = scores[row] * deg_inv[row]
            new_scores = torch.zeros(num_nodes, dtype=torch.float32)
            new_scores.scatter_add_(0, col, messages)
            new_scores = self.alpha * new_scores + (1 - self.alpha) * personalization
            if torch.norm(new_scores - scores, p=1) < tol:
                break
            scores = new_scores

        return scores
