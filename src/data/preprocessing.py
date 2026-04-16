import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["item_id", "user_id", "rating", "timestamp"],
    )
    return df

def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["year"] = df["datetime"].dt.year
    return df

def filter_min_interactions(df, min_user=5, min_item=5):
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()

    df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
    df = df[df["item_id"].isin(item_counts[item_counts >= min_item].index)]

    return df