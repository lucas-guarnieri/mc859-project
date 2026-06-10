import os
import networkx as nx
import pandas as pd


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    """Build an undirected bipartite graph from an interaction DataFrame.

    Each edge connects a user node to an item node and carries
    timestamp and rating as attributes.
    """
    G = nx.Graph()
    edges = [
        (
            int(row.user_id),
            int(row.item_id),
            {"timestamp": str(row.timestamp), "rating": float(row.rating)},
        )
        for row in df.itertuples(index=False)
    ]
    G.add_edges_from(edges)
    return G


def save_graph(G: nx.Graph, path: str) -> None:
    """Save a graph to GraphML format, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_graphml(G, path)
