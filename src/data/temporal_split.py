import os
import pandas as pd


def create_snapshots(df: pd.DataFrame, output_dir: str) -> None:
    """Write one cumulative snapshot CSV per year found in the dataset.

    Each file snapshot_YYYY.csv contains all interactions up to and
    including that year.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = df.sort_values("timestamp")
    years = sorted(df["year"].unique())
    print(f"Generating {len(years)} snapshots...")
    for year in years:
        snapshot = df[df["year"] <= year]
        path = os.path.join(output_dir, f"snapshot_{year}.csv")
        snapshot.to_csv(path, index=False)
        print(f"  {path} ({len(snapshot)} rows)")
    print("Done.")
