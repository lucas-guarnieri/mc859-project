import networkx as nx
import pandas as pd


def encode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte user_id e item_id para inteiros (reduz memória)
    """

    df["user_id"] = df["user_id"].astype("category").cat.codes
    df["item_id"] = df["item_id"].astype("category").cat.codes

    df["item_id"] += df["user_id"].nunique()

    return df


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Constrói grafo bipartido user-item
    """

    G = nx.Graph()

    G.add_edges_from(zip(df["user_id"], df["item_id"]))

    return G


def save_graph(G: nx.Graph, path: str):
    """
    Salva grafo em formato GraphML (exigência da entrega)
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    nx.write_graphml(G, path)