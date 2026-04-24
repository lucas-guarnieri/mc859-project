import pickle
from pathlib import Path
from src.utils.io import load_config

config = load_config("configs/base.yaml")




def get_num_users():
    with open(config["data"]["user_map_path"], "rb") as f:
        user_map = pickle.load(f)
    return len(user_map)


NUM_USERS = get_num_users()


def is_user(node_id) -> bool:
    return int(node_id) < NUM_USERS


def is_product(node_id) -> bool:
    return int(node_id) >= NUM_USERS