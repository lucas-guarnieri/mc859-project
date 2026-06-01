import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import norm as sparse_norm
from src.graph.graph_utils import get_product_nodes, get_user_nodes, get_user_products
 
 
class ItemCFRecommender:
 
    name = "item_cf"
 
    def __init__(self, top_n_similar: int = 50):
        # number of most similar items to store per product
        self.top_n_similar = top_n_similar
        self.item_similarity: dict = {}  # product -> [(product, similarity), ...]
        self.product_list: list = []     # ordered list of products (index reference)
 
    def fit(self, G: nx.Graph):
        product_nodes = sorted(get_product_nodes(G))
        user_nodes = sorted(get_user_nodes(G))
 
        product_index = {p: i for i, p in enumerate(product_nodes)}
        user_index = {u: i for i, u in enumerate(user_nodes)}
 
        n_products = len(product_nodes)
        n_users = len(user_nodes)
 
        # Build sparse rating matrix M (products x users)
        # M[i, j] = rating given by user j to product i
        M = lil_matrix((n_products, n_users), dtype=np.float32)
        for u, v, data in G.edges(data=True):
            user    = u if u in user_index else v
            product = v if u in user_index else u
            if user in user_index and product in product_index:
                rating = float(data.get("rating", 1.0))
                M[product_index[product], user_index[user]] = rating
 
        # Convert to CSR for efficient row operations
        M = M.tocsr()
 
        # L2-normalize each product vector (row)
        row_norms = np.array(M.power(2).sum(axis=1)).flatten() ** 0.5
        row_norms[row_norms == 0] = 1e-10
        # divide each row by its norm
        from scipy.sparse import diags
        D = diags(1.0 / row_norms)
        M_norm = D @ M
 
        # Compute cosine similarity row by row to avoid materializing full matrix
        # For each product, compute dot product with all others and keep top-N
        self.item_similarity = {}
        self.product_list = product_nodes
 
        for i, product in enumerate(product_nodes):
            # dot product of product i with all other products
            sims = (M_norm @ M_norm[i].T).toarray().flatten()
            sims[i] = 0.0  # exclude self-similarity
 
            # keep only top-N similar products
            top_idx = np.argpartition(sims, -self.top_n_similar)[-self.top_n_similar:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
 
            self.item_similarity[product] = [
                (product_nodes[j], float(sims[j]))
                for j in top_idx if sims[j] > 0
            ]
 
        # M and M_norm are local and will be garbage collected
        return self
 
    def recommend(self, G: nx.Graph, user, k: int = 10) -> list:
        seen = set(get_user_products(G, user))
        scores = defaultdict(float)
 
        # Get ratings the user gave to seen items
        user_ratings = {
            nbr: float(G[user][nbr].get("rating", 1.0))
            for nbr in seen
        }
 
        # Aggregate similarity scores weighted by the user's own ratings
        for item in seen:
            for similar_item, sim in self.item_similarity.get(item, []):
                if similar_item not in seen:
                    scores[similar_item] += sim * user_ratings[item]
 
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:k]]