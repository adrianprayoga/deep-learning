import os
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from training_config import TrainingConfig
import torch
import random

class TrainingDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.root_dir = self.config.train_dataset
        self.resolution = self.config.resolution
        self.augment = self.config.augment_data
        self.dataset_size = self.config.dataset_size
        self.image_paths = {0: [], 1: []}
        self.labels = []

        if self.config.train_dataset == 'mixed':
            mixed_datasets = {
                '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_train': 0.5,
                '/home/ginger/code/gderiddershanghai/deep-learning/data/hyperreenact': 0.2,
                '/home/ginger/code/gderiddershanghai/deep-learning/data/fsgan': 0.2,
                '/home/ginger/code/gderiddershanghai/deep-learning/data/CollabDiff': 0.1,
            }

            for root_dir, ratio in mixed_datasets.items():
                subset_size = int(self.dataset_size * ratio) // 2

                for label, subdir in enumerate(['real', 'fake']):
                    subdir_path = os.path.join(root_dir, subdir)
                    if not os.path.exists(subdir_path):
                        continue

                    files = [
                        os.path.join(dirpath, file)
                        for dirpath, _, filenames in os.walk(subdir_path)
                        for file in filenames if file.lower().endswith((".jpg", ".png"))
                    ]

                    if len(files) >= subset_size:
                        sampled_files = random.sample(files, subset_size)
                    else:
                        sampled_files = files
                    
                    self.image_paths[label].extend(sampled_files)

        if self.config.train_dataset != 'mixed':
            for label, subdir in enumerate(['real', 'fake']):
                subdir_path = os.path.join(self.root_dir, subdir)
                if not os.path.exists(subdir_path):
                    raise ValueError(f"Subdirectory '{subdir}' not found in {self.root_dir}.")

                for dirpath, _, files in os.walk(subdir_path):
                    for file in files:
                        if file.lower().endswith((".jpg", ".png")):
                            self.image_paths[label].append(os.path.join(dirpath, file))

        if not any(self.image_paths.values()):
            raise ValueError(f"No .jpg or .png files found in {self.root_dir}.")

        min_size = min(len(self.image_paths[0]), len(self.image_paths[1]))
        max_size_per_class = min(self.dataset_size // 2, min_size)

        self.image_paths[0] = self.image_paths[0][:max_size_per_class]
        self.image_paths[1] = self.image_paths[1][:max_size_per_class]

        self.labels = [0] * len(self.image_paths[0]) + [1] * len(self.image_paths[1])
        self.image_paths = self.image_paths[0] + self.image_paths[1]

        self.transform = self._init_data_aug_method() if self.augment else self._init_basic_preprocessing()

    def _init_data_aug_method(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-10, 10), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.OneOf([
                A.Resize(height=self.resolution, width=self.resolution, interpolation=cv2.INTER_CUBIC),
                A.Resize(height=self.resolution, width=self.resolution, interpolation=cv2.INTER_LINEAR),
                A.Resize(height=self.resolution, width=self.resolution, interpolation=cv2.INTER_AREA),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(-0.1, 0.1),
                    p=0.5
                ),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            ], p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
            ToTensorV2(),
        ])

    def _init_basic_preprocessing(self):
        return A.Compose([
            A.Resize(height=self.resolution, width=self.resolution, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Error loading image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=image)
            image_tensor = transformed["image"].float()

        except Exception as e:
            return self.__getitem__((index + 1) % len(self.image_paths))

        return image_tensor, label, image_path

if __name__ == "__main__":
    config = TrainingConfig()
    dataset = TrainingDataset(config=config)
    print(f"Number of images: {len(dataset)}")

    for i in range(3):
        image_tensor, label, image_path = dataset[i]
        print(f"Sample {i}: Image shape: {image_tensor.shape}, Label: {label}, Path: {image_path}")