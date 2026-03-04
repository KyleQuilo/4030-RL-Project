import yaml

def load_config(path="config.yaml"):
    """Load YAML config file into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
