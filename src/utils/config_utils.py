import os
from pathlib import Path

import torch
import yaml

def load_model_config(model, weights_path=None):
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    with open(os.path.join(PROJECT_DIR, 'src/config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    if model == 'spsl':
        with open(os.path.join(PROJECT_DIR, 'src/config/spsl.yaml'), 'r') as f:
            model_config = yaml.safe_load(f)
        model_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

        if weights_path is None:
            model_config['pretrained'] = os.path.join(PROJECT_DIR, config["weights_path"])
        else:
            model_config['pretrained'] = weights_path

        print(model_config)

        return model_config

    return None