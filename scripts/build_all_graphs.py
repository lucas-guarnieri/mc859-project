from pathlib import Path
import pandas as pd

from src.graph.build_graph import build_bipartite_graph, save_graph
from src.utils.io import load_config

config = load_config("configs/base.yaml")

def main():
    input_dir = Path(config["data"]["snapshots_dir"])
    output_dir = Path(config["data"]["graphs_dir"])
    selected_years = config["graph"]["years"]

    files = sorted(input_dir.glob("snapshot_*.csv"))
    for file in files:

        year = int(file.stem.split("_")[1])
        if selected_years is not None and year not in selected_years:
            continue

        print(f"Processing {year}...")

        df = pd.read_csv(file)
        G = build_bipartite_graph(df)
        output_path = output_dir / f"graph_{year}.graphml"
        save_graph(G, str(output_path))

        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()  