from dataclasses import dataclass
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

@dataclass
class InferenceDataset(Dataset):
    """
    A PyTorch Dataset class for loading .jpg images for inference.
    """
    root_dir: str  # Root directory containing subdirectories of images
    resolution: int = 224  # Desired resolution for resizing images

    def __post_init__(self):
        """
        Initializes the dataset by collecting all .jpg image paths.
        """
        # Collect all .jpg files in the specified directory and its subdirectories
        self.image_paths = [
            os.path.join(dirpath, file)
            for dirpath, _, files in os.walk(self.root_dir)
            for file in files if file.lower().endswith(".jpg")
        ]

        if not self.image_paths:
            raise ValueError(f"No .jpg files found in {self.root_dir}.")

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
            str: The original file path of the image.
        """
        # Get the file path for the image
        image_path = self.image_paths[index]

        # Load the image using PIL
        image = Image.open(image_path).convert("RGB")

        # Resize the image
        image = image.resize((self.resolution, self.resolution), Image.ANTIALIAS)

        # Convert the image to a tensor (C, H, W format)
        image_tensor = torch.tensor(
            data=np.array(image).transpose(2, 0, 1),  # (H, W, C) -> (C, H, W)
            dtype=torch.float32,
        ) / 255.0  # Normalize to [0, 1]

        return image_tensor, image_path

if __name__ == "__main__":
    # Example for testing the dataset with the given directories
    # root_dir = "/home/ginger/code/gderiddershanghai/deep-learning/data/starganv2"
    root_dir = "/home/ginger/code/gderiddershanghai/deep-learning/data/MidJourney"

    resolution = 224 #clip
    # resolution = 256 #xception #and spls

    dataset = InferenceDataset(root_dir=root_dir, resolution=resolution)

    print(f"Number of images: {len(dataset)}")

    # Test loading an image
    image_tensor, image_path = dataset[0]
    print(f"Image shape: {image_tensor.shape}, Path: {image_path}")
