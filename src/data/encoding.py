import pandas as pd


def create_global_mappings(df: pd.DataFrame):
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}

    return user_map, item_map


def apply_mappings(df: pd.DataFrame, user_map, item_map):
    df = df.copy()

    df["user_id"] = df["user_id"].map(user_map)

    df["item_id"] = df["item_id"].map(item_map)
    df["item_id"] += len(user_map)

    df["user_id"] = df["user_id"].astype("int32")
    df["item_id"] = df["item_id"].astype("int32")

    return df