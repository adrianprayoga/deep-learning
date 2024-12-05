import os
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from training_config import TrainingConfig  # Assuming the TrainingConfig file is named training_config.py


@dataclass
class InferenceDataset(Dataset):
    """
    A PyTorch Dataset class for loading images for inference.
    """
    config: TrainingConfig  # Configuration object
    use_val_set1: bool = True  # If True, use val_dataset1; otherwise, use val_dataset2

    def __post_init__(self):
        """
        Initializes the dataset by collecting all image paths and their labels.
        """
        self.root_dir = self.config.val_dataset1 if self.use_val_set1 else self.config.val_dataset2
        self.resolution = self.config.resolution
        self.model_name = self.config.model_name

        self.image_paths = []
        self.labels = []

        # Traverse subdirectories to collect images and assign labels
        for label, subdir in enumerate(['real', 'fake']):  # 0 for real, 1 for fake
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.exists(subdir_path):
                raise ValueError(f"Subdirectory '{subdir}' not found in {self.root_dir}.")
            
            for dirpath, _, files in os.walk(subdir_path):
                for file in files:
                    if file.lower().endswith((".jpg", ".png")):
                        self.image_paths.append(os.path.join(dirpath, file))
                        self.labels.append(label)  # Assign label based on subdirectory

        if not self.image_paths:
            raise ValueError(f"No .jpg or .png files found in {self.root_dir}.")

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """
        Loads and processes an image at the given index.

        Args:
            index (int): Index of the image to load.

        Returns:
            torch.Tensor: The loaded image as a PyTorch tensor.
            int: The label of the image (0 for real, 1 for fake).
        """
        # Get the file path and label for the image
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Load the image using PIL
        image = Image.open(image_path).convert("RGB")

        # Resize the image
        image = image.resize((self.resolution, self.resolution), Image.ANTIALIAS)
        original_image = np.array(image)

        # Convert the image to a tensor (C, H, W format)
        image_tensor = torch.tensor(
            data=np.array(image).transpose(2, 0, 1),  # (H, W, C) -> (C, H, W)
            dtype=torch.float32,
        ) / 255.0  # Normalize to [0, 1]

        # Normalize using CLIP's expected mean and std
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor, label, image_path, original_image


if __name__ == "__main__":
    # Load the configuration
    config = TrainingConfig()

    # Example: Use val_dataset1
    dataset = InferenceDataset(config=config, use_val_set1=True)
    print(f"Number of images in {config.val_dataset1_name}: {len(dataset)}")

    # Test loading an image and label
    image_tensor, label, image_path, original_image = dataset[-1]
    print(f"Image shape: {image_tensor.shape}, Label: {label}, Path: {image_path}")

    # Example: Use val_dataset2
    dataset2 = InferenceDataset(config=config, use_val_set1=False)
    print(f"Number of images in {config.val_dataset2_name}: {len(dataset2)}")
