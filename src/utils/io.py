import os
import yaml
import pickle

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config("configs/base.yaml")
MODELS_DIR = config["results"]["models_dir"]

def save_model(model, graph_name: str):
    year = graph_name.split("_")[-1]
    filename = f"{model.name}_{year}.pkl"
    path = os.path.join(MODELS_DIR, filename)
 
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
 
 
def load_model(model_class, graph_name: str):
    year = graph_name.split("_")[-1]
    filename = f"{model_class.name}_{year}.pkl"
    path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at {path}. Run fit() and save_model() first.")
 
    with open(path, "rb") as f:
        return pickle.load(f)