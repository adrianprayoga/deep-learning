from dataclasses import dataclass
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

# @dataclass
# class InferenceDataset(Dataset):
#     """
#     A PyTorch Dataset class for loading .jpg images for inference.
#     """
#     root_dir: str  # Root directory containing subdirectories of images
#     resolution: int = 224  # Desired resolution for resizing images

#     def __post_init__(self):
#         """
#         Initializes the dataset by collecting all .jpg image paths.
#         """
#         # Collect all .jpg files in the specified directory and its subdirectories
#         self.image_paths = []
#         self.labels = []

#         # Traverse subdirectories to collect images and assign labels
#         for label, subdir in enumerate(['real', 'fake']):  # 0 for real, 1 for fake
#             subdir_path = os.path.join(self.root_dir, subdir)
#             if not os.path.exists(subdir_path):
#                 raise ValueError(f"Subdirectory '{subdir}' not found in {self.root_dir}.")
            
#             for dirpath, _, files in os.walk(subdir_path):
#                 for file in files:
#                     if file.lower().endswith(".jpg"):
#                         self.image_paths.append(os.path.join(dirpath, file))
#                         self.labels.append(label)  # Assign label based on subdirectory

#         if not self.image_paths:
#             raise ValueError(f"No .jpg files found in {self.root_dir}.")

#     def __len__(self) -> int:
#         """
#         Returns the total number of images in the dataset.
#         """
#         return len(self.image_paths)

#     def __getitem__(self, index: int):
#         """
#         Loads and processes an image at the given index.

#         Args:
#             index (int): Index of the image to load.

#         Returns:
#             torch.Tensor: The loaded image as a PyTorch tensor.
#             str: The original file path of the image.
#         """
#         # Get the file path for the image
#         image_path = self.image_paths[index]

#         # Load the image using PIL
#         image = Image.open(image_path).convert("RGB")

#         # Resize the image
#         image = image.resize((self.resolution, self.resolution), Image.ANTIALIAS)

#         # Convert the image to a tensor (C, H, W format)
#         image_tensor = torch.tensor(
#             data=np.array(image).transpose(2, 0, 1),  # (H, W, C) -> (C, H, W)
#             dtype=torch.float32,
#         ) / 255.0  # Normalize to [0, 1]

#         # # Normalize using CLIP's expected mean and std
#         mean = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)
#         std = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)
#         image_tensor = (image_tensor - mean) / std


#         return image_tensor, image_path


@dataclass
class InferenceDataset(Dataset):
    """
    A PyTorch Dataset class for loading .jpg images for inference.
    """
   
    root_dir: str  # Root directory containing subdirectories of images
    resolution: int   # Desired resolution for resizing images
    model_name: str
     
    def __post_init__(self):
        """
        Initializes the dataset by collecting all .jpg image paths and their labels.
        """
        self.resolution = 224 if 'clip' in self.model_name.lower() else 256

        self.image_paths = []
        self.labels = []

        # Traverse subdirectories to collect images and assign labels
        for label, subdir in enumerate(['real', 'fake']):  # 0 for real, 1 for fake
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.exists(subdir_path):
                raise ValueError(f"Subdirectory '{subdir}' not found in {self.root_dir}.")
            
            for dirpath, _, files in os.walk(subdir_path):
                for file in files:
                    if file.lower().endswith(".jpg"):
                        self.image_paths.append(os.path.join(dirpath, file))
                        self.labels.append(label)  # Assign label based on subdirectory

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
    # Example for testing the dataset with the given directories
    root_dir = "/home/ginger/code/gderiddershanghai/deep-learning/data/MidJourney"
    model_name = "clip_DF40"
    resolution = 224  # CLIP resolution
    dataset = InferenceDataset(root_dir=root_dir, resolution=resolution,model_name = model_name)

    print(f"Number of images: {len(dataset)}")

    # Test loading an image and label
    image_tensor, label, image_path = dataset[-1]

    print(f"Image shape: {image_tensor.shape}, Label: {label}")
