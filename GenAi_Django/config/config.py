import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
