from pathlib import Path

from src.analysis.graph_metrics import (
    load_graph,
    compute_basic_metrics,
    get_degree_list,
    plot_degree_distribution,
    compute_components,
    basic_component_metrics,
    plot_component_distribution,
)
from src.utils.io import load_config
from src.graph.graph_utils import is_user, is_product


def print_metrics(name, metrics):
    print(f"\n=== {name.upper()} ===")
    print("Vertices:", metrics["vertices"])
    if metrics["edges"] is not None:
        print("Edges:", metrics["edges"])
    print("Average degree:", round(metrics["avg_degree"], 4))


def main():
    config = load_config("configs/base.yaml")
    graph_path = config["data"]["graph2018_path"]
    out_dir = Path(config["results"]["metrics_plot_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graph...")
    G = load_graph(graph_path)

    all_nodes = list(G.nodes())
    user_nodes    = [n for n in all_nodes if is_user(n)]
    product_nodes = [n for n in all_nodes if is_product(n)]

    print("\n===== BASIC METRICS =====")

    full_metrics = compute_basic_metrics(G)
    print_metrics("Full Graph", full_metrics)
    plot_degree_distribution(
        get_degree_list(G),
        "Degree Distribution - Full Graph",
        out_dir / "degree_full.png",
    )

    user_metrics = compute_basic_metrics(G, user_nodes)
    print_metrics("Users", user_metrics)
    plot_degree_distribution(
        get_degree_list(G, user_nodes),
        "Degree Distribution - Users",
        out_dir / "degree_users.png",
    )

    product_metrics = compute_basic_metrics(G, product_nodes)
    print_metrics("Products", product_metrics)
    plot_degree_distribution(
        get_degree_list(G, product_nodes),
        "Degree Distribution - Products",
        out_dir / "degree_products.png",
    )

    print(f"\nPlots saved to {out_dir}/")

    print("\n===== COMPONENT ANALYSIS =====")
    _, sizes = compute_components(G)
    component_metrics = basic_component_metrics(sizes)
    print("Number of components:", component_metrics["num_components"])
    print("Largest component:", component_metrics["largest_component"])
    print("Smallest component:", component_metrics["smallest_component"])
    print("Average size:", round(component_metrics["avg_size"], 4))
    plot_component_distribution(
        sizes,
        "Connected Component Size Distribution",
        out_dir / "component_distribution.png",
    )
    print(f"\nPlot saved to {out_dir}/component_distribution.png")


if __name__ == "__main__":
    main()
