import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


def load_graph(path: str) -> nx.Graph:
    G = nx.read_graphml(path)
    G = nx.relabel_nodes(G, lambda x: int(x))

    return G


def get_degree_list(G: nx.Graph, nodes=None):
    if nodes is None:
        return [d for _, d in G.degree()]
    return [G.degree(n) for n in nodes]


def compute_basic_metrics(G: nx.Graph, nodes=None):
    if nodes is None:
        num_vertices = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = (2 * num_edges) / num_vertices

        return {
            "vertices": num_vertices,
            "edges": num_edges,
            "avg_degree": avg_degree
        }

    degrees = get_degree_list(G, nodes)

    return {
        "vertices": len(nodes),
        "edges": None,
        "avg_degree": sum(degrees) / len(nodes)
    }


def plot_degree_distribution(
    degrees,
    title: str,
    output_path: str
):
    count = Counter(degrees)

    x = sorted(count.keys())
    y = [count[k] for k in x]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=12, alpha=0.7)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_components(G):
    components = list(nx.connected_components(G))
    sizes = [len(c) for c in components]

    return components, sizes


def basic_component_metrics(sizes):
    return {
        "num_components": len(sizes),
        "largest_component": max(sizes),
        "smallest_component": min(sizes),
        "avg_size": sum(sizes) / len(sizes),
    }


def plot_component_distribution(
    sizes,
    title,
    output_path
):
    count = Counter(sizes)

    x = sorted(count.keys())
    y = [count[k] for k in x]

    plt.figure(figsize=(8,5))
    plt.scatter(x, y, s=12, alpha=0.7)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Component size (k)")
    plt.ylabel("Number of components")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()