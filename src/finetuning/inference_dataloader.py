# import os
# from dataclasses import dataclass
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from training_config import TrainingConfig  # Assuming the TrainingConfig file is named training_config.py


# @dataclass
# class InferenceDataset(Dataset):
#     """
#     A PyTorch Dataset class for loading images for inference.
#     """
#     config: TrainingConfig  # Configuration object
#     use_val_set1: bool = True  # If True, use val_dataset1; otherwise, use val_dataset2

#     def __post_init__(self):
#         """
#         Initializes the dataset by collecting all image paths and their labels.
#         """
#         self.root_dir = self.config.val_dataset1 if self.use_val_set1 else self.config.val_dataset2
#         self.resolution = self.config.resolution
#         self.model_name = self.config.model_name

#         self.image_paths = []
#         self.labels = []

#         # Traverse subdirectories to collect images and assign labels
#         for label, subdir in enumerate(['real', 'fake']):  # 0 for real, 1 for fake
#             subdir_path = os.path.join(self.root_dir, subdir)
#             if not os.path.exists(subdir_path):
#                 raise ValueError(f"Subdirectory '{subdir}' not found in {self.root_dir}.")
            
#             for dirpath, _, files in os.walk(subdir_path):
#                 for file in files:
#                     if file.lower().endswith((".jpg", ".png")):
#                         self.image_paths.append(os.path.join(dirpath, file))
#                         self.labels.append(label)  # Assign label based on subdirectory

#         if not self.image_paths:
#             raise ValueError(f"No .jpg or .png files found in {self.root_dir}.")

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
#             int: The label of the image (0 for real, 1 for fake).
#         """
#         # Get the file path and label for the image
#         image_path = self.image_paths[index]
#         label = self.labels[index]

#         # Load the image using PIL
#         image = Image.open(image_path).convert("RGB")

#         # Resize the image
#         image = image.resize((self.resolution, self.resolution), Image.ANTIALIAS)
#         original_image = np.array(image)

#         # Convert the image to a tensor (C, H, W format)
#         image_tensor = torch.tensor(
#             data=np.array(image).transpose(2, 0, 1),  # (H, W, C) -> (C, H, W)
#             dtype=torch.float32,
#         ) / 255.0  # Normalize to [0, 1]

#         # Normalize using CLIP's expected mean and std
#         mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
#         std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
#         image_tensor = (image_tensor - mean) / std

#         return image_tensor, label, image_path, original_image


# if __name__ == "__main__":
#     # Load the configuration
#     config = TrainingConfig()

#     # Example: Use val_dataset1
#     dataset = InferenceDataset(config=config, use_val_set1=True)
#     print(f"Number of images in {config.val_dataset1_name}: {len(dataset)}")

#     # Test loading an image and label
#     image_tensor, label, image_path, original_image = dataset[-1]
#     print(f"Image shape: {image_tensor.shape}, Label: {label}, Path: {image_path}")

#     # Example: Use val_dataset2
#     dataset2 = InferenceDataset(config=config, use_val_set1=False)
#     print(f"Number of images in {config.val_dataset2_name}: {len(dataset2)}")
import os
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from training_config import TrainingConfig


@dataclass
class InferenceDataset(Dataset):
    """
    A PyTorch Dataset class for loading balanced 'real' and 'fake' images.
    """
    config: TrainingConfig  # Configuration object
    dataset_index: int = 1  # Dataset selector: 1, 2, or 3

    def __post_init__(self):
        """
        Initializes the dataset by collecting image paths and balancing the sizes of real and fake images.
        """
        # Select the correct validation dataset based on the provided index
        dataset_map = {
            1: (self.config.val_dataset1, self.config.val_dataset1_name),
            2: (self.config.val_dataset2, self.config.val_dataset2_name),
            3: (self.config.val_dataset3, self.config.val_dataset3_name),
        }
        if self.dataset_index not in dataset_map:
            raise ValueError(f"Invalid dataset index {self.dataset_index}. Must be 1, 2, or 3.")

        self.root_dir, self.dataset_name = dataset_map[self.dataset_index]
        self.resolution = self.config.resolution
        self.image_paths = []
        self.labels = []

        # Collect images and labels for 'real' and 'fake'
        real_images, fake_images = self._collect_images()

        # Balance the sizes of the two lists
        min_size = min(len(real_images), len(fake_images))
        balanced_images = real_images[:min_size] + fake_images[:min_size]
        balanced_labels = [0] * min_size + [1] * min_size

        # Shuffle the balanced dataset
        combined = list(zip(balanced_images, balanced_labels))
        np.random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

        if not self.image_paths:
            raise ValueError(f"No .jpg or .png files found in {self.root_dir}.")

    def _collect_images(self):
        """
        Collects 'real' and 'fake' images from corresponding subdirectories.

        Returns:
            tuple: Lists of real and fake image paths.
        """
        real_images, fake_images = [], []
        for label, subdir in enumerate(['real', 'fake']):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.exists(subdir_path):
                raise ValueError(f"Subdirectory '{subdir}' not found in {self.root_dir}.")
            image_list = real_images if label == 0 else fake_images

            for dirpath, _, files in os.walk(subdir_path):
                for file in files:
                    if file.lower().endswith((".jpg", ".png")):
                        image_list.append(os.path.join(dirpath, file))
        return real_images, fake_images

    def __len__(self) -> int:
        """
        Returns the total number of balanced images in the dataset.
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
            str: The file path of the loaded image.
            np.ndarray: The original image as a NumPy array (for debugging).
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.resolution, self.resolution), Image.ANTIALIAS)
            original_image = np.array(image)

            # Convert the image to a tensor (C, H, W format)
            image_tensor = torch.tensor(
                np.array(image).transpose(2, 0, 1), dtype=torch.float32
            ) / 255.0  # Normalize to [0, 1]

            # Normalize using CLIP's expected mean and std
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        except (IOError, ValueError, RuntimeError) as e:
            print(f"Error loading image {image_path}: {e}")

            # Return a blank tensor and invalid label if loading fails
            image_tensor = torch.zeros((3, self.resolution, self.resolution), dtype=torch.float32)
            original_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            label = -1  # Use -1 for corrupted or invalid data
            image_path = "corrupted"

        return image_tensor, label, image_path, original_image



    # def __getitem__(self, index: int):
    #     """
    #     Loads and processes an image at the given index.

    #     Args:
    #         index (int): Index of the image to load.

    #     Returns:
    #         torch.Tensor: The loaded image as a PyTorch tensor.
    #         int: The label of the image (0 for real, 1 for fake).
    #     """
    #     image_path = self.image_paths[index]
    #     label = self.labels[index]

    #     # Load and preprocess the image
    #     image = Image.open(image_path).convert("RGB")
    #     image = image.resize((self.resolution, self.resolution), Image.ANTIALIAS)
    #     original_image = np.array(image)

    #     # Convert the image to a tensor (C, H, W format)
    #     image_tensor = torch.tensor(
    #         np.array(image).transpose(2, 0, 1), dtype=torch.float32
    #     ) / 255.0  # Normalize to [0, 1]

    #     # Normalize using CLIP's expected mean and std
    #     mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    #     std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    #     image_tensor = (image_tensor - mean) / std

    #     return image_tensor, label, image_path, original_image


if __name__ == "__main__":
    import torch

    num_workers = torch.multiprocessing.cpu_count()
    print(f"Using {num_workers} workers.")

    config = TrainingConfig()
    config.val_dataset1 = '/home/ginger/code/gderiddershanghai/deep-learning/data/CollabDiff'
    config.val_dataset2 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random'
    config.val_dataset3 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_train'

    # Example: Use val_dataset1
    dataset1 = InferenceDataset(config=config, dataset_index=1)
    print(f"Number of images in {config.val_dataset1_name}: {len(dataset1)}")

    # Test loading an image and label
    image_tensor, label, image_path, original_image = dataset1[-1]
    print(f"Image shape: {image_tensor.shape}, Label: {label}, Path: {image_path}")

    # Example: Use val_dataset2
    dataset2 = InferenceDataset(config=config, dataset_index=2)
    print(f"Number of images in {config.val_dataset2_name}: {len(dataset2)}")

    # Example: Use val_dataset3
    dataset3 = InferenceDataset(config=config, dataset_index=3)
    print(f"Number of images in {config.val_dataset3_name}: {len(dataset3)}")



# if __name__ == "__main__":
#     # Load the configuration
#     import torch
#     num_workers = torch.multiprocessing.cpu_count()
#     print(f"Using {num_workers} workers.")

#     num_workers = torch.multiprocessing.cpu_count()

#     config = TrainingConfig()
#     config.val_dataset1 = '/home/ginger/code/gderiddershanghai/deep-learning/data/CollabDiff'
#     config.val_dataset2 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random'
#     config.val_dataset3 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_train'
#     # Example: Use val_dataset1
#     dataset1 = InferenceDataset(config=config, dataset_index=1)
#     print(f"Number of images in CD {config.val_dataset1_name}: {len(dataset1)}")

#     # Test loading an image and label
#     image_tensor, label, image_path, original_image = dataset1[-1]
#     print(f"Image shape: random {image_tensor.shape}, Label: {label}, Path: {image_path}")

#     # Example: Use val_dataset2
#     dataset2 = InferenceDataset(config=config, dataset_index=2)
#     print(f"Number of images in random {config.val_dataset2_name}: {len(dataset2)}")

#     # Example: Use val_dataset3
#     dataset3 = InferenceDataset(config=config, dataset_index=3)
#     print(f"Number of images in full {config.val_dataset3_name}: {len(dataset3)}")
