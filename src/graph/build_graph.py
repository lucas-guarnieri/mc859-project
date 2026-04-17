import os
import networkx as nx
import pandas as pd


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(zip(df["user_id"], df["item_id"]))

    return G


def save_graph(G: nx.Graph, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_graphml(G, path)