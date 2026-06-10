import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV with columns item_id, user_id, rating, timestamp."""
    return pd.read_csv(
        path,
        header=None,
        names=["item_id", "user_id", "rating", "timestamp"],
    )


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Unix timestamp and add datetime and year columns."""
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["year"] = df["datetime"].dt.year
    return df


def filter_min_year(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    """Keep only interactions from min_year onwards."""
    return df[df["year"] >= min_year]


def filter_min_interactions(
    df: pd.DataFrame, min_user: int, min_item: int
) -> pd.DataFrame:
    """Remove users and items below the minimum interaction count thresholds."""
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()
    df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
    df = df[df["item_id"].isin(item_counts[item_counts >= min_item].index)]
    return df
