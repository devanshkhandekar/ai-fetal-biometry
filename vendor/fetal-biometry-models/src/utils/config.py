from pathlib import Path
import yaml


def load_yaml(path):
    path = Path(path)
    with open(path, 'r') as f:
        return yaml.safe_load(f)
