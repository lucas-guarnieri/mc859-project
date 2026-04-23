from src.data.preprocessing import (load_raw_data, add_datetime, filter_min_year,filter_min_interactions,)
from src.data.encoding import create_global_mappings, apply_mappings
from src.data.temporal_split import create_snapshots
from src.utils.io import load_config

import pickle



config = load_config("configs/base.yaml")

df = load_raw_data(config["data"]["raw_path"])
df = add_datetime(df)
df = filter_min_year(df, min_year=config["filtering"]["min_year"])
df = filter_min_interactions(
    df,
    min_user=config["filtering"]["min_user_interactions"],
    min_item=config["filtering"]["min_item_interactions"]
)

user_map, item_map = create_global_mappings(df)
with open(config["data"]["user_map_path"], "wb") as f:
    pickle.dump(user_map, f)
with open(config["data"]["item_map_path"], "wb") as f:
    pickle.dump(item_map, f)
df = apply_mappings(df, user_map, item_map)

df.to_csv(config["data"]["processed_path"], index=False) 

create_snapshots(df, config["data"]["snapshots_dir"])