import os
import csv
import numpy as np
from sklearn import metrics
from training_config import TrainingConfig
from inference_dataloader import InferenceDataset

def get_test_metrics(
    y_pred, y_true, config, dataset_name, save=False
):
    """
    Calculates evaluation metrics for testing and logs them to a CSV file.

    Args:
        y_pred (np.ndarray): Predicted probabilities for each image.
        y_true (np.ndarray): Ground truth labels for each image.
        config (TrainingConfig): Configuration object.
        dataset_name (str): Name of the validation dataset (e.g., "MidJourney").
        save (bool): Whether to save metrics to a CSV file. Default is False.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = np.squeeze(y_pred)
    # print(os.path.basename(config.train_dataset), '1232312312312')
    # Frame-level metrics
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    ap = metrics.average_precision_score(y_true, y_pred)

    # Accuracy
    prediction_class = (y_pred > 0.5).astype(int)
    acc = (prediction_class == y_true).mean()

    if save:
        # Append metrics to the CSV file
        file_exists = os.path.exists(config.csv_path)
        with open(config.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Epoch", "Training Loss","Accuracy", "AUC", "EER", "AP", "LoRA", "LoRA Rank (r)", "LoRA Alpha",
                    "LoRA Dropout", "Number of Target Modules", "Size of Dataset","Augmentation", "Model", "Learning Rate", 
                    "Optimizer", "Train Dataset", "Validation Dataset"
                ])
            if config.lora_config.target_modules: 
                lora_size = len(config.lora_config.target_modules)
                # print('target modes', config.lora_config.target_modules)
            
            else: lora_size=None
            writer.writerow([
                config.current_epoch, config.loss, acc, auc, eer, ap,
                config.use_lora, config.lora_config.r, config.lora_config.lora_alpha,
                config.lora_config.lora_dropout, lora_size, config.dataset_size,
                config.augment_data, config.model_name, config.learning_rate, config.optimizer,
                os.path.basename(config.train_dataset), dataset_name
            ])

        print(f"Metrics for epoch {config.current_epoch} saved to {config.csv_path}")

    return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}


if __name__ == "__main__":
    # Load the configuration
    config = TrainingConfig()

    # Example: Use val_dataset1
    dataset1 = InferenceDataset(config=config, use_val_set1=True)
    print(f"Number of images in {config.val_dataset1_name}: {len(dataset1)}")

    # Example: Use val_dataset2
    dataset2 = InferenceDataset(config=config, use_val_set1=False)
    print(f"Number of images in {config.val_dataset2_name}: {len(dataset2)}")

    # Simulated predictions and labels for testing
    y_pred = np.random.rand(len(dataset1))  # Random predictions between 0 and 1
    y_true = np.random.randint(0, 2, len(dataset1))  # Random ground truth labels (0 or 1)

    # Compute metrics for val_dataset1
    metrics1 = get_test_metrics(
        y_pred=y_pred,
        y_true=y_true,
        config=config,
        dataset_name=config.val_dataset1_name,
        save=True
    )
    print('---------------------FAKE FAKE FAKE---------------------------')
    print(f"FAKE FAKE FAKE Metrics for {config.val_dataset1_name}: {metrics1}")
    print('---------------------FAKE FAKE FAKE---------------------------')

    # Simulated predictions and labels for val_dataset2
    y_pred = np.random.rand(len(dataset2))
    y_true = np.random.randint(0, 2, len(dataset2))

    # Compute metrics for val_dataset2
    metrics2 = get_test_metrics(
        y_pred=y_pred,
        y_true=y_true,
        config=config,
        dataset_name=config.val_dataset2_name,
        save=True
    )
    print('---------------------FAKE FAKE FAKE---------------------------')

    print(f"FAKE FAKE FAKE Metrics for {config.val_dataset2_name}: {metrics2}")
    print('---------------------FAKE FAKE FAKE---------------------------')
