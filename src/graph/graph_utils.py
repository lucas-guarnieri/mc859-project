import pickle
import networkx as nx
from pathlib import Path
from src.utils.io import load_config

config = load_config("configs/base.yaml")




_NUM_USERS = None


def _get_num_users() -> int:
    global _NUM_USERS
    if _NUM_USERS is None:
        with open(config["data"]["user_map_path"], "rb") as f:
            _NUM_USERS = len(pickle.load(f))
    return _NUM_USERS


def is_user(node_id) -> bool:
    return int(node_id) < _get_num_users()


def is_product(node_id) -> bool:
    return int(node_id) >= _get_num_users()


def get_user_nodes(G: nx.Graph) -> list:
    return [n for n in G.nodes() if is_user(n)]

def get_product_nodes(G: nx.Graph) -> list:
    return [n for n in G.nodes() if is_product(n)]

def get_user_products(G: nx.Graph, user) -> list:
    return list(G.neighbors(user))

def get_product_users(G: nx.Graph, product) -> list:
    return list(G.neighbors(product))

def get_ratings(G: nx.Graph, user) -> dict:
    return {
        nbr: float(G[user][nbr].get("rating", 0.0))
        for nbr in G.neighbors(user)
    }