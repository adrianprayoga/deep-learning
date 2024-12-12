import os
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from training_config import TrainingConfig

@dataclass
class InferenceDataset(Dataset):
    config: TrainingConfig
    dataset_index: int = 1

    def __post_init__(self):
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

        real_images, fake_images = self._collect_images()

        min_size = min(len(real_images), len(fake_images))
        balanced_images = real_images[:min_size] + fake_images[:min_size]
        balanced_labels = [0] * min_size + [1] * min_size

        combined = list(zip(balanced_images, balanced_labels))
        np.random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

        if not self.image_paths:
            raise ValueError(f"No .jpg or .png files found in {self.root_dir}.")

    def _collect_images(self):
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
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.resolution, self.resolution), Image.ANTIALIAS)
            original_image = np.array(image)

            image_tensor = torch.tensor(
                np.array(image).transpose(2, 0, 1), dtype=torch.float32
            ) / 255.0

            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        except (IOError, ValueError, RuntimeError) as e:
            print(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros((3, self.resolution, self.resolution), dtype=torch.float32)
            original_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            label = -1
            image_path = "corrupted"

        return image_tensor, label, image_path, original_image

if __name__ == "__main__":
    import torch

    num_workers = torch.multiprocessing.cpu_count()
    print(f"Using {num_workers} workers.")

    config = TrainingConfig()
    config.val_dataset1 = '/home/ginger/code/gderiddershanghai/deep-learning/data/CollabDiff'
    config.val_dataset2 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random'
    config.val_dataset3 = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_train'

    dataset1 = InferenceDataset(config=config, dataset_index=1)
    print(f"Number of images in {config.val_dataset1_name}: {len(dataset1)}")

    image_tensor, label, image_path, original_image = dataset1[-1]
    print(f"Image shape: {image_tensor.shape}, Label: {label}, Path: {image_path}")

    dataset2 = InferenceDataset(config=config, dataset_index=2)
    print(f"Number of images in {config.val_dataset2_name}: {len(dataset2)}")

    dataset3 = InferenceDataset(config=config, dataset_index=3)
    print(f"Number of images in {config.val_dataset3_name}: {len(dataset3)}")