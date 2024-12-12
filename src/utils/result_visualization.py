import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def visualize_predictions(all_preds, all_labels, all_img_names, dataset):
    """
    Visualizes predictions with the following:
    1. Class distribution.
    2. Misclassified samples.
    3. Probability distribution for each class.

    Args:
        all_preds (np.array): Predicted probabilities for the "fake" class.
        all_labels (np.array): Ground truth labels (0 for real, 1 for fake).
        all_img_names (list): List of image paths.
        dataset (InferenceDataset): The dataset object for loading images.
    """
    # Convert probabilities to predicted classes
    predicted_classes = (all_preds > 0.5).astype(int)

    # Class-wise counts
    real_count = np.sum(all_labels == 0)
    fake_count = np.sum(all_labels == 1)
    print(f"Real samples: {real_count}, Fake samples: {fake_count}")

    # Confusion matrix-like analysis
    correct_real = np.sum((predicted_classes == 0) & (all_labels == 0))
    correct_fake = np.sum((predicted_classes == 1) & (all_labels == 1))
    incorrect_real = np.sum((predicted_classes == 1) & (all_labels == 0))
    incorrect_fake = np.sum((predicted_classes == 0) & (all_labels == 1))

    print(f"Correct Real: {correct_real}, Correct Fake: {correct_fake}")
    print(f"Incorrect Real: {incorrect_real}, Incorrect Fake: {incorrect_fake}")

    # Plot class-wise distribution
    plt.figure()
    plt.bar(["Correct Real", "Correct Fake", "Incorrect Real", "Incorrect Fake"],
            [correct_real, correct_fake, incorrect_real, incorrect_fake])
    plt.title("Class-wise Prediction Distribution")
    plt.ylabel("Count")
    plt.show()

    # Plot misclassified images
    misclassified_indices = np.where(predicted_classes != all_labels)[0]
    if misclassified_indices.size > 0:
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(misclassified_indices[:8]):  # Show up to 8 misclassified samples
            image_path = all_img_names[idx]
            image = Image.open(image_path).convert("RGB")
            plt.subplot(2, 4, i + 1)
            plt.imshow(image)
            plt.title(f"True: {all_labels[idx]}, Pred: {predicted_classes[idx]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


