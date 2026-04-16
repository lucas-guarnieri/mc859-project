from src.data.preprocessing import (
    load_raw_data,
    add_datetime,
    filter_min_interactions,
)
from src.data.encoding import create_global_mappings, apply_mappings

from src.data.temporal_split import create_snapshots
from src.utils.io import load_config

config = load_config("configs/base.yaml")

df = load_raw_data(config["data"]["raw_path"])
df = add_datetime(df)

df = df.sample(500000, random_state=42).sort_values("timestamp")

df = filter_min_interactions(
    df,
    min_user=config["filtering"]["min_user_interactions"],
    min_item=config["filtering"]["min_item_interactions"]
)

# sanity check
assert df.isnull().sum().sum() == 0

df.to_csv(config["data"]["processed_sample_path"], index=False)

user_map, item_map = create_global_mappings(df)

df = apply_mappings(df, user_map, item_map)

create_snapshots(df, config["data"]["snapshots_dir"])

# debug
print("Shape final:", df.shape)

print("\nDistribuição por ano:")
print(df["year"].value_counts().sort_index())

print("\nUsuários únicos:", df["user_id"].nunique())
print("Itens únicos:", df["item_id"].nunique())