from pathlib import Path
import pandas as pd

from src.graph.build_graph import build_bipartite_graph, save_graph


def main():

    input_dir = Path("data/snapshots")
    output_dir = Path("data/graphs")

    files = sorted(input_dir.glob("snapshot_*.parquet"))

    for file in files:

        year = file.stem.split("_")[1]

        print(f"Processando {year}...")

        df = pd.read_parquet(file)

        G = build_bipartite_graph(df)

        output_path = output_dir / f"graph_{year}.graphml"

        save_graph(G, str(output_path))

        print(f"Salvo em {output_path}")


if __name__ == "__main__":
    main()