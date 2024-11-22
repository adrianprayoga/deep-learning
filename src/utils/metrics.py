import numpy as np
import pandas as pd
from sklearn import metrics


def get_test_metrics(
    y_pred, y_true, img_names, model_name, training_ds, testing_ds, save=False
):
    """
    Calculates evaluation metrics for testing on .jpg images.

    Args:
        y_pred (np.ndarray): Predicted probabilities for each image.
        y_true (np.ndarray): Ground truth labels for each image.
        img_names (list): List of image file paths or names.
        model_name (str): Name of the model used for predictions.
        training_ds (str): Name of the training dataset.
        testing_ds (str): Name of the testing dataset.
        save (bool): Whether to save predictions to a DataFrame. Default is False.

    Returns:
        dict: Dictionary containing evaluation metrics:
            - acc: Accuracy
            - auc: Frame-level AUC
            - eer: Frame-level EER
            - ap: Average Precision
            - pred: Predictions
            - label: Ground truth labels
            - total: Total number of predictions
    """
    y_pred = np.squeeze(y_pred)

    # Frame-level metrics
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    ap = metrics.average_precision_score(y_true, y_pred)

    # Frame-level accuracy
    prediction_class = (y_pred > 0.5).astype(int)
    acc = (prediction_class == y_true).sum() / len(y_true)

    # Save results to a DataFrame if the save flag is True
    if save:
        results_df = pd.DataFrame({
            "Image": img_names,
            "Prediction": y_pred,
            "True_Label": y_true,
            "Model": [model_name] * len(y_pred),
            "Training_Dataset": [training_ds] * len(y_pred),
            "Testing_Dataset": [testing_ds] * len(y_pred),
        })

        # Save to a CSV file
        csv_filename = f"outputs/{model_name}_{training_ds}_to_{testing_ds}_predictions.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"Predictions saved to {csv_filename}")

    return {
        "acc": acc,
        "auc": auc,
        "eer": eer,
        "ap": ap,
        "pred": y_pred,
        "label": y_true,
        "total": len(y_pred),
    }


def test_get_test_metrics():
    # Synthetic test data
    y_pred = np.array([0.1, 0.4, 0.8, 0.9, 0.3, 0.6])
    y_true = np.array([0, 0, 1, 1, 0, 1])
    img_names = [
        "image1.jpg", "image2.jpg", "image3.jpg",
        "image4.jpg", "image5.jpg", "image6.jpg"
    ]

    # Model and dataset info
    model_name = "MyModel"
    training_ds = "TrainingDataset"
    testing_ds = "TestingDataset"

    # Call the function with save=True to test saving predictions
    metrics_dict = get_test_metrics(
        y_pred=y_pred,
        y_true=y_true,
        img_names=img_names,
        model_name=model_name,
        training_ds=training_ds,
        testing_ds=testing_ds,
        save=True
    )

    # Display results
    print("Test Metrics:")
    for key, value in metrics_dict.items():
        if isinstance(value, (float, int)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    test_get_test_metrics()
