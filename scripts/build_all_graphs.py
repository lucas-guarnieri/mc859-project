from pathlib import Path
import pandas as pd

from src.graph.build_graph import build_bipartite_graph, save_graph
from src.utils.io import load_config


def main():
    config = load_config("configs/base.yaml")
    input_dir = Path(config["data"]["snapshots_dir"])
    output_dir = Path(config["data"]["graphs_dir"])
    selected_years = config["graph"]["years"]

    for file in sorted(input_dir.glob("snapshot_*.csv")):
        year = int(file.stem.split("_")[1])
        if selected_years is not None and year not in selected_years:
            continue
        print(f"Building graph for {year}...")
        G = build_bipartite_graph(pd.read_csv(file))
        output_path = output_dir / f"graph_{year}.graphml"
        save_graph(G, str(output_path))
        print(f"  Saved {output_path}")


if __name__ == "__main__":
    main()
