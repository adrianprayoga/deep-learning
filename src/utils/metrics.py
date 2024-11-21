import os
import pandas as pd

def save_predictions_to_csv(predictions, img_names, model_name, training_ds, testing_ds):
    """
    Save predictions to a CSV file in the outputs directory.

    Args:
        predictions (list or np.ndarray): List of predictions.
        img_names (list): List of image file names.
        model_name (str): Name of the model.
        training_ds (str): Name of the training dataset.
        testing_ds (str): Name of the testing dataset.
    """
    # Define the CSV filename in the outputs directory
    outputs_dir = "outputs"
    csv_filename = os.path.join(outputs_dir, f"{model_name}_{training_ds}_to_{testing_ds}_predictions.csv")

    # Create a DataFrame
    results_df = pd.DataFrame({
        "Image": img_names,
        "Prediction": predictions,
        "Model": [model_name] * len(predictions),
        "Training_Dataset": [training_ds] * len(predictions),
        "Testing_Dataset": [testing_ds] * len(predictions),
    })

    # Save to CSV
    results_df.to_csv(csv_filename, index=False)
    print(f"Predictions saved to {csv_filename}")


if __name__ == "__main__":
    test_get_test_metrics()
