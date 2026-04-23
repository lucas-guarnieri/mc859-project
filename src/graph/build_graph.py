import os
import networkx as nx
import pandas as pd


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    edges = [
        (
            int(row.user_id),
            int(row.item_id),
            {
                "timestamp": str(row.timestamp),
                "rating": float(row.rating)
            }
        )
        for row in df.itertuples(index=False)
    ]

    G.add_edges_from(edges)

    return G


def save_graph(G: nx.Graph, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_graphml(G, path)