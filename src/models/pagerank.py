import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import GDC
from collections import defaultdict
from src.graph.graph_utils import get_product_nodes, get_user_products
 
 
class PersonalizedPageRankRecommender:
 
    name = "pagerank"
 
    def __init__(self, alpha: float = 0.85, top_k_ppr: int = 500):
        """
        alpha:      teleport probability back to the source node
        top_k_ppr:  number of top PPR nodes to consider per user
        """
        self.alpha = alpha
        self.top_k_ppr = top_k_ppr
        self.pyg_data = None         # converted PyG graph
        self.node_list: list = []    # ordered node list (index <-> node id)
        self.node_index: dict = {}   # node id -> index in node_list
        self.product_set: set = set()
 
    def fit(self, G: nx.Graph):
        # Store ordered node list for index mapping
        self.node_list = list(G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.node_list)}
        self.product_set = set(get_product_nodes(G))
 
        # Convert NetworkX graph to PyG format
        # Nodes are relabeled to 0..N-1 internally
        self.pyg_data = from_networkx(G)
        self.pyg_data.num_nodes = G.number_of_nodes()
 
        return self
 
    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
 
        # Get index of the user node in the PyG graph
        user_idx = self.node_index[user]
 
        # Build personalization vector: 1.0 at user node, 0 elsewhere
        num_nodes = len(self.node_list)
        personalization = torch.zeros(num_nodes, dtype=torch.float32)
        personalization[user_idx] = 1.0
 
        # Compute PPR scores using power iteration
        edge_index = self.pyg_data.edge_index
        ppr_scores = self._compute_ppr(edge_index, personalization, num_nodes)
 
        # Rank unseen products by PPR score
        candidates = []
        for p in self.product_set:
            if p not in seen:
                idx = self.node_index[p]
                candidates.append((p, float(ppr_scores[idx])))
 
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
        """
        Power iteration for Personalized PageRank.
        
        PPR(t+1) = alpha * A_norm * PPR(t) + (1 - alpha) * personalization
        
        where A_norm is the row-normalized adjacency matrix.
        """
        # Compute degree for normalization
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.float32))
        deg_inv = 1.0 / deg.clamp(min=1.0)
 
        # Initialize PPR scores from personalization
        scores = personalization.clone()
 
        for _ in range(max_iter):
            # Propagate: for each edge (u -> v), pass scores[u] / deg[u] to v
            messages = scores[row] * deg_inv[row]
            new_scores = torch.zeros(num_nodes, dtype=torch.float32)
            new_scores.scatter_add_(0, col, messages)
 
            # Teleport
            new_scores = self.alpha * new_scores + (1 - self.alpha) * personalization
 
            # Check convergence
            if torch.norm(new_scores - scores, p=1) < tol:
                break
 
            scores = new_scores
 
        return scores