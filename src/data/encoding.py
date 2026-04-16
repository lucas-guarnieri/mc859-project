import pandas as pd


def create_global_mappings(df: pd.DataFrame):
    """
    Cria mapeamento global consistente para usuários e itens
    """

    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}

    return user_map, item_map


def apply_mappings(df: pd.DataFrame, user_map, item_map):
    """
    Aplica mapeamento consistente
    """

    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    df["item_id"] += len(user_map)

    return df