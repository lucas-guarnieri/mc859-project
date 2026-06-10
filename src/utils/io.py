import yaml


def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
