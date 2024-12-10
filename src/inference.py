import os
import time
from pathlib import Path

import torch
import yaml

from utils.model_loading import get_detector
from utils.dataloader import InferenceDataset
from utils.testing import evaluate_model_with_metrics

if __name__ == '__main__':

    ## FILE THAT NEEDS TO BE UPDATED
    # Change the file below to your local config name
    # run python inference.py from main folder e.g. python src/inference.py
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    with open(os.path.join(PROJECT_DIR, 'src/config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    dataset_eval = InferenceDataset(root_dir=config['data_root_dir'], resolution=224, model_name=config['model_type'])
    detector = get_detector(config['model_type'],
                            load_weights=True,
                            weights_path=config["weights_path"])

    if detector is not None:
        results = evaluate_model_with_metrics(
            detector=detector,
            dataset_eval=dataset_eval,
            training_ds=config['training_ds'],
            testing_ds=config['testing_ds'],
            model_name=config['model_type'],
            batch_size=16,
            save=False  # Save predictions to a CSV
        )

