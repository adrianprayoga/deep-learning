import os
import torch
from peft import LoraConfig

class TrainingConfig:
    """
    Configuration class to manage training settings and hyperparameters.
    """
    def __init__(self):
        # Paths
        self.model_weights = '/home/ginger/code/gderiddershanghai/deep-learning/weights/clip_DF40/clip.pth'
        self.train_dataset = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random_hegyncollab_reals"
        self.val_dataset1 = "/home/ginger/code/gderiddershanghai/deep-learning/data/MidJourney"
        self.val_dataset1_name = os.path.basename(self.val_dataset1)
        self.val_dataset2 = "/home/ginger/code/gderiddershanghai/deep-learning/data/starganv2"
        self.val_dataset2_name = os.path.basename(self.val_dataset2)
        self.csv_path = "/home/ginger/code/gderiddershanghai/deep-learning/outputs/finetune_results/training_metrics_JDB_random_hegyncollab_realsv2.csv"
        self.checkpoint_dir = "/home/ginger/code/gderiddershanghai/deep-learning/weights_finetuned"
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Dataset settings
        self.dataset_size = 2000

        # Model settings

        self.use_lora = True  # Whether LoRA is applied
        self.model_name = "CLIP_LoRA" if self.use_lora else "CLIP_Full"
        self.augment_data = True  # Whether data augmentation is applied
        self.lora_config = LoraConfig(
        # task_type=TaskType.FEATURE_EXTRACTION, 
        r=16, 
        lora_alpha=64, 
        target_modules=None, 
        lora_dropout=0.1, 
        bias="none"
        )

        # Training hyperparameters
        self.learning_rate = 2e-3 if self.use_lora else 1e-6
        self.batch_size = 32
        self.epochs = 10
        self.current_epoch = 0
        self.loss = None
        self.optimizer = "AdamW"
        self.loss_function = "CrossEntropyLoss"

        # Resolution and device
        self.resolution = 224  # Image resolution (e.g., 224x224 for CLIP)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Reproducibility
        self.seed = 42
        
    def save_config(self, path=None):
        """
        Save the configuration to a JSON file for reproducibility.
        """
        import json
        if path is None:
            path = f"config_{self.model_name}_{self.dataset_size}.json"
        config_dict = {key: value for key, value in self.__dict__.items() if not callable(value)}
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to {path}")
        
if __name__ == "__main__":
    config = TrainingConfig()
    print(config.val_dataset1_name)
    print(config.val_dataset2_name)
    config.save_config("training_config.json")
