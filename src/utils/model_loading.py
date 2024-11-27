import torch
import torch.nn as nn
from transformers import CLIPModel
from torchvision import models
import timm

class CLIPDetector(nn.Module):
    """
    CLIP-based detector for binary classification (real vs fake).
    Designed for inference with CLIP Base (ViT-B/16).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Load the CLIP Base visual backbone
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
        
        # Classification head for binary classification
        self.head = nn.Linear(768, num_classes)  # 768 is the hidden size for ViT-B/16

    def forward(self, image_tensor):
        """
        Forward pass for inference.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Probabilities for each class (real and fake).
        """
        # Extract features using the backbone
        features = self.backbone(image_tensor)['pooler_output']

        # Classify features
        logits = self.head(features)

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

class XceptionDetector(nn.Module):
    """
    Xception-based detector for binary classification (real vs fake).
    Optimized for inference.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Load the pretrained Xception model
        self.backbone = timm.create_model('xception', pretrained=True)

        # Replace the classifier head for binary classification
        num_features = self.backbone.get_classifier().in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, image_tensor):
        """
        Forward pass for inference.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Probabilities for each class (real and fake).
        """
        # Pass through the model
        logits = self.backbone(image_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
