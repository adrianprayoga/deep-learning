import os
import torch
from peft import LoraConfig

class TrainingConfig:
    def __init__(self):
        # Paths
        self.model_weights = '/home/ginger/code/gderiddershanghai/deep-learning/weights/clip_DF40/clip.pth'
        # self.train_dataset = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random"
        self.train_dataset = "mixed"
        self.val_dataset1 = "/home/ginger/code/gderiddershanghai/deep-learning/data/MidJourney"
        self.val_dataset1_name = os.path.basename(self.val_dataset1)
        self.val_dataset2 = "/home/ginger/code/gderiddershanghai/deep-learning/data/starganv2"
        self.val_dataset2_name = os.path.basename(self.val_dataset2)
        self.val_dataset3 = "/home/ginger/code/gderiddershanghai/deep-learning/data/heygen"
        self.val_dataset3_name = os.path.basename(self.val_dataset3)
        
        self.csv_path = f"/home/ginger/code/gderiddershanghai/deep-learning/outputs/finetune_results/FINE_metrics_{os.path.basename(self.train_dataset)}.csv"
        self.checkpoint_dir = "/home/ginger/code/gderiddershanghai/deep-learning/weights_finetuned"
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.dataset_size = 200

        # Model settings

        self.use_lora = True  
        self.model_name = "CLIP_LoRA" if self.use_lora else "CLIP_Full"
        self.augment_data = True  
        self.lora_config = LoraConfig(
        # task_type=TaskType.FEATURE_EXTRACTION, 
        r=16, 
        lora_alpha=32, 
        target_modules=None, 
        lora_dropout=0.2, 
        bias="none"
        )

        self.learning_rate = 1e-3 if self.use_lora else 5e-6
        self.batch_size = 32
        self.epochs = 5
        self.current_epoch = 0
        self.loss = None
        self.optimizer = "AdamW"
        self.loss_function = "CrossEntropyLoss"
        self.use_weight_decay = False
        self.weight_decay = 0.01

        self.resolution = 224  # 224x224 for CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        
        
    def update_hyperparams(self):

        self.csv_path = f"/home/ginger/code/gderiddershanghai/deep-learning/outputs/finetune_results/FINE_metrics_{os.path.basename(self.train_dataset)}.csv"
        

        if self.dataset_size <= 500:
            print("Warning: Dataset size is very small; overfitting risk is high.")
            
            if self.use_lora:
                self.learning_rate = 1.5e-3  
                self.lora_config.lora_alpha = 16  
                self.lora_config.lora_dropout = 0.3  
                self.use_weight_decay = False
                self.weight_decay = 0.0
            else: 
                self.learning_rate = 3e-4 
                self.use_weight_decay = True
                self.weight_decay = 0.05  

        else:
            if self.use_lora:
                if self.dataset_size < 2000:
                    self.learning_rate = 9e-4  
                    self.lora_config.lora_alpha = 16
                elif self.dataset_size <= 10000:
                    self.learning_rate = 4e-4  
                    self.lora_config.lora_alpha = 32
                else:
                    self.learning_rate = 1.8e-4  
                    self.lora_config.r = 16
                    self.lora_config.lora_alpha = 64
                    self.lora_config.lora_dropout = 0.1

                self.use_weight_decay = False
                self.weight_decay = 0.0

            else: 
                if self.dataset_size <= 2000:
                    self.learning_rate = 4e-5  
                elif self.dataset_size <= 10000:
                    self.learning_rate = 2e-5  
                else:
                    self.learning_rate = 1e-5  

                self.use_weight_decay = True
                self.weight_decay = 0.01

        print(f" - Learning Rate: {self.learning_rate}")
        print(f" - Use Weight Decay: {self.use_weight_decay}")
        print(f" - Weight Decay: {self.weight_decay}")
        print(f" - Dataset Size: {self.dataset_size}")

        if self.use_lora:
            print(f" - LoRA Alpha: {self.lora_config.lora_alpha}")
            print(f" - LoRA Rank (r): {self.lora_config.r}")
            print(f" - LoRA Dropout: {self.lora_config.lora_dropout}")


    def save_config(self, path=None):

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
