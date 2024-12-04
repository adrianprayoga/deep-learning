import torch
import torch.nn as nn
from transformers import CLIPModel
from torchvision import models
import timm
from src.detectors.spsl_detector import SpslDetector

def get_detector(type, num_classes=2, load_weights=False, weights_path=None, device='cpu', config=None):
    model = None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ## XCEPTION NOT TESTED!!!
    if type == 'xception':
        model = XceptionDetector(num_classes=num_classes)
        if load_weights:
            if weights_path is None:
                print('Please check, something wrong as we are trying to load weights but path is None')
                return None

            missing_keys, unexpected_keys = None, None
            if weights_path is not None:
                # Load pre-trained weights
                state_dict = torch.load(weights_path, map_location=device)
                # Adjust for any prefixes in the state_dict keys (e.g., 'module.')
                if any(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                state_dict['backbone.fc.weight'] = state_dict.pop('backbone.last_linear.weight')
                state_dict['backbone.fc.bias'] = state_dict.pop('backbone.last_linear.bias')

                for key in list(state_dict.keys()):
                    if "adjust_channel" in key:
                        # print('removing', key, 'as it is unused')
                        state_dict.pop(key)

                # Load the state dictionary with relaxed strictness
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Optional: Log or handle missing/unexpected keys
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
    elif type == 'clip':
        model = CLIPDetector(num_classes=2)
        if load_weights:
            if weights_path is None:
                print('Please check, something wrong as we are trying to load weights but path is None')
                return None

            missing_keys, unexpected_keys = None, None
            if weights_path is not None:
                # Load pre-trained weights
                state_dict = torch.load(weights_path, map_location=device)
                # Adjust for any prefixes in the state_dict keys (e.g., 'module.')
                if any(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # Load the state dictionary with relaxed strictness
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Optional: Log or handle missing/unexpected keys
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
    elif type == 'spsl':
        if config is None:
            print('require config')
            return None
        model = SpslDetector(config, load_weights=True)

    return model


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
    def __init__(self, num_classes=2, load_weights=False):
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
