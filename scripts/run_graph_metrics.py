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
from src.utils.node_types import is_user, is_product


config = load_config("configs/base.yaml")

GRAPH_PATH = config["data"]["graph2018_path"]
OUT_DIR = Path(config["results"]["metrics_plot_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)


def print_metrics(name, metrics):
    print(f"\n=== {name.upper()} ===")
    print("Vertices:", metrics["vertices"])

    if metrics["edges"] is not None:
        print("Edges:", metrics["edges"])

    print("Average degree:", round(metrics["avg_degree"], 4))


def main():
    print("Loading graph...")
    G = load_graph(GRAPH_PATH)

    all_nodes = list(G.nodes())
    user_nodes = [n for n in all_nodes if is_user(n)]
    product_nodes = [n for n in all_nodes if is_product(n)]

    print("\n===== BASIC METRICS =====")

    # --------------------------
    # FULL GRAPH
    # --------------------------
    full_metrics = compute_basic_metrics(G)
    print_metrics("Full Graph", full_metrics)

    full_degrees = get_degree_list(G)

    plot_degree_distribution(
        full_degrees,
        "Degree Distribution - Full Graph",
        OUT_DIR / "degree_full.png",
    )

    # --------------------------
    # USERS
    # --------------------------
    user_metrics = compute_basic_metrics(G, user_nodes)
    print_metrics("Users", user_metrics)

    user_degrees = get_degree_list(G, user_nodes)

    plot_degree_distribution(
        user_degrees,
        "Degree Distribution - Users",
        OUT_DIR / "degree_users.png",
    )

    # --------------------------
    # PRODUCTS
    # --------------------------
    product_metrics = compute_basic_metrics(G, product_nodes)
    print_metrics("Products", product_metrics)

    product_degrees = get_degree_list(G, product_nodes)

    plot_degree_distribution(
        product_degrees,
        "Degree Distribution - Products",
        OUT_DIR / "degree_products.png",
    )

    print(f"\nPlots saved to {OUT_DIR}/")


    # --------------------------
    # COMPONENTS
    # --------------------------
    print("\n===== COMPONENT ANALYSIS =====")

    print("Computing connected components...")
    _, sizes = compute_components(G)

    component_metrics = basic_component_metrics(sizes)

    print("\n=== COMPONENTS ===")
    print("Number of components:", component_metrics["num_components"])
    print("Largest component:", component_metrics["largest_component"])
    print("Smallest component:", component_metrics["smallest_component"])
    print("Average size:", round(component_metrics["avg_size"], 4))

    plot_component_distribution(
        sizes,
        "Connected Component Size Distribution",
        OUT_DIR / "component_distribution.png"
    )

    print("\nPlot saved to results/plots/component_distribution.png")


if __name__ == "__main__":
    main()