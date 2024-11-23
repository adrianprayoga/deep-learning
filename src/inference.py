import time
import torch
import yaml

from detectors.spsl_detector import SpslDetector
from utils.dataloader import InferenceDataset
from utils.model_loading import CLIPDetector
from utils.testing import evaluate_model_with_metrics

if __name__ == '__main__':

    with open('adrian_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset_eval = InferenceDataset(root_dir=config['data_root_dir'], resolution=224)

    detector = None
    if config['model_type'] == "spsl":
        with open('config/spsl.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        model_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        detector = SpslDetector(model_config)
        model_name = "SPSL"
        weights_path=None
    elif config['model_type'] == "clip":
        detector = CLIPDetector(num_classes=2)
        weights_path = config["weights_path"]
        model_name = "CLIP_Base"
    else:
        print('model type is not supported!!!')

    if detector is not None:
        results = evaluate_model_with_metrics(
            detector=detector,
            weights_path=weights_path,
            dataset_eval=dataset_eval,
            model_name=model_name,
            training_ds=config['training_ds'],
            testing_ds=config['testing_ds'],
            batch_size=16,
            save=True  # Save predictions to a CSV
        )