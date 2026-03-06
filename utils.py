import yaml

def load_config(path="config.yaml"):
    """
    Load a YAML configuration file into a Python dictionary.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
