import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import get_peft_model, LoraConfig, TaskType

class CLIPFineTuneDetector(nn.Module):
    """
    CLIP-based detector for fine-tuning with LoRA support.
    Most of the code here is the same as in the original CLIP Model File
    """
    def __init__(self, num_classes=2, apply_lora=False, lora_config=None):
        super().__init__()
        
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
        self.head = nn.Linear(768, num_classes)

        # Apply LoRA if specified
        if apply_lora and lora_config is not None:
            self.backbone = get_peft_model(self.backbone, lora_config)

    def forward(self, image_tensor):
        """
        Forward pass for fine-tuning.
        """
        # Extract features using the backbone
        features = self.backbone(image_tensor)['pooler_output']
        
        # Classify features
        logits = self.head(features)

        return logits
