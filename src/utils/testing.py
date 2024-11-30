from .dataloader import InferenceDataset 
from .model_loading import CLIPDetector, XceptionDetector
from .metrics import get_test_metrics
import torch
from torch.utils.data import DataLoader
import numpy as np
from .result_visualization import visualize_predictions
import ClassVisualization


@torch.no_grad()
def evaluate_model_with_metrics(
    detector, weights_path, dataset_eval,
    model_name, training_ds, testing_ds, batch_size=16, device=None, save=False):
    """
    Evaluates a CLIP-based detector on a specified evaluation dataset with metrics calculation.

    Args:
        detector (torch.nn.Module): The model to be evaluated.
        weights_path (str): Path to the pre-trained model weights.
        dataset_eval (InferenceDataset): Custom dataset for evaluation.
        model_name (str): Name of the model used for predictions.
        training_ds (str): Name of the training dataset.
        testing_ds (str): Name of the testing dataset.
        batch_size (int): Batch size for inference. Default is 16.
        device (str): Device to use for inference. Default is "cuda" if available, else "cpu".
        save (bool): Whether to save predictions to a CSV file. Default is False.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    # Set the device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    missing_keys, unexpected_keys = None, None
    if weights_path is not None:
    # Load pre-trained weights
        state_dict = torch.load(weights_path, map_location=device)
    
        # Adjust for any prefixes in the state_dict keys (e.g., 'module.')
        if 'clip' in model_name.lower():
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load the state dictionary with relaxed strictness
        missing_keys, unexpected_keys = detector.load_state_dict(state_dict, strict=False)

    # Optional: Log or handle missing/unexpected keys
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
        detector.eval().to(device)
        
    print('loaded the dictionary')
    detector = detector.to(device)
    detector.eval()

    print('loaded the model')

    # Prepare the evaluation DataLoader
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)

    # Collect predictions and labels
    all_preds = []
    all_labels = []
    all_img_names = []
    i=0
    with torch.no_grad():
        for batch in dataloader_eval:
            if i%15 ==0: print(f'batch {i}')
            i+=1
            images, labels, image_paths = batch
            images = images.to(device)

            # Run the model
            detector.eval()
            probabilities = detector(images)
            all_preds.extend(probabilities[:, 1].cpu().numpy())  # Fake class probability
            all_labels.extend(labels.numpy())  # Collect true labels
            all_img_names.extend(image_paths)
            # print('probs', probabilities)
            # print('labels', labels)
            # break
            

    print('finished eval')
    # Convert predictions and labels to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print('all_preds ', all_preds )
    print('all labels', all_labels)

    # visualize
    visualize_predictions(all_preds, all_labels, all_img_names, dataset_eval)

    # Calculate metrics
    metrics_result = get_test_metrics(
        y_pred=all_preds,
        y_true=all_labels,
        img_names=all_img_names,
        model_name=model_name,
        training_ds=training_ds,
        testing_ds=testing_ds,
        save=save
    )
    print(f"Test Metrics for {training_ds}_{testing_ds}:")
    for key, value in metrics_result.items():
        if isinstance(value, (float, int)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    return metrics_result

if __name__ == "__main__":
    resolution = 224 #224
    testing_ds = "MidJourney"
    # training_ds = "clip_DF40"
    training_ds = "exception_DF40"
    root_dir = f"/home/ginger/code/gderiddershanghai/deep-learning/data/{testing_ds}" #starganv2 #MidJourney
    dataset_eval = InferenceDataset(root_dir=root_dir, resolution=resolution,model_name = training_ds)

    # detector = CLIPDetector(num_classes=2)
    detector = XceptionDetector(num_classes=2)
    # weights_path = f"/home/ginger/code/gderiddershanghai/deep-learning/weights/{training_ds}/clip.pth"
    weights_path = 'weights/exception_DF40/xception.pth'

    results = evaluate_model_with_metrics(
        detector=detector,
        weights_path=weights_path,
        dataset_eval=dataset_eval,
        model_name= training_ds,
        training_ds=training_ds, # "FaceSwap" #FaceReenactment $ EntireFaceSynthesis # DF40
        testing_ds = testing_ds,
        batch_size=16,
        save=True  # Save predictions to a CSV
)
