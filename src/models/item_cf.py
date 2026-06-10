import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import lil_matrix, diags
from src.graph.graph_utils import get_product_nodes, get_user_nodes, get_user_products


class ItemCFRecommender:
    """Item-based collaborative filtering using cosine similarity.

    Builds a sparse products×users rating matrix, L2-normalizes each row,
    and stores the top-N most similar products per item to bound memory.
    Recommendation score is the similarity-weighted sum of the user's ratings.
    """

    name = "item_cf"
    tracks_paths = False

    def __init__(self, top_n_similar: int = 50):
        self.top_n_similar = top_n_similar
        self.item_similarity: dict = {}
        self.product_list: list = []

    def fit(self, G: nx.Graph):
        product_nodes = sorted(get_product_nodes(G))
        user_nodes = sorted(get_user_nodes(G))

        product_index = {p: i for i, p in enumerate(product_nodes)}
        user_index = {u: i for i, u in enumerate(user_nodes)}

        n_products = len(product_nodes)
        n_users = len(user_nodes)

        M = lil_matrix((n_products, n_users), dtype=np.float32)
        for u, v, data in G.edges(data=True):
            user = u if u in user_index else v
            product = v if u in user_index else u
            if user in user_index and product in product_index:
                M[product_index[product], user_index[user]] = float(data.get("rating", 1.0))

        M = M.tocsr()
        row_norms = np.array(M.power(2).sum(axis=1)).flatten() ** 0.5
        row_norms[row_norms == 0] = 1e-10
        M_norm = diags(1.0 / row_norms) @ M

        self.item_similarity = {}
        self.product_list = product_nodes

        for i, product in enumerate(product_nodes):
            sims = (M_norm @ M_norm[i].T).toarray().flatten()
            sims[i] = 0.0
            top_idx = np.argpartition(sims, -self.top_n_similar)[-self.top_n_similar:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            self.item_similarity[product] = [
                (product_nodes[j], float(sims[j]))
                for j in top_idx if sims[j] > 0
            ]

        return self

    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
        scores = defaultdict(float)
        user_ratings = {nbr: float(G[user][nbr].get("rating", 1.0)) for nbr in seen}
        for item in seen:
            for similar_item, sim in self.item_similarity.get(item, []):
                if similar_item not in seen:
                    scores[similar_item] += sim * user_ratings[item]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:k]]
