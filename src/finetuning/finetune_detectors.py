import torch
from peft import get_peft_model
from transformers import CLIPModel
import torch.nn as nn
from training_config import TrainingConfig  
import timm

# # https://github.com/huggingface/peft/issues/1988

# """
# Xception needs to be updated - didn't check after I created a training_config file
# """

class XceptionDetector(nn.Module):
    """
    Xception-based detector for binary classification (real vs fake).
    Optimized for inference.
    """
    def __init__(self, num_classes=2, weights_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.device = device

        # Load the pretrained Xception model
        self.backbone = timm.create_model("xception", pretrained=False)

        # Replace the classifier head for binary classification
        num_features = self.backbone.get_classifier().in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

        # Load custom pretrained weights if specified
        if weights_path:
            self._load_pretrained_weights(weights_path)

    def forward(self, image_tensor):
        """
        Forward pass for inference.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Probabilities for each class (real and fake).
        """
        logits = self.backbone(image_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def _load_pretrained_weights(self, weights_path):
        """
        Load and align pretrained weights for inference.
        """
        try:
            state_dict = torch.load(weights_path, map_location=self.device)

            # Correct key mismatches by removing prefixes
            corrected_state_dict = {}
            for k, v in state_dict.items():
                # Remove prefixes `module.` and `backbone.`
                new_key = k.replace("module.", "").replace("backbone.", "")
                
                # Handle classifier name difference
                if new_key == "last_linear.weight":
                    new_key = "fc.weight"
                elif new_key == "last_linear.bias":
                    new_key = "fc.bias"

                corrected_state_dict[new_key] = v

            # Load the corrected state dictionary
            missing_keys, unexpected_keys = self.backbone.load_state_dict(corrected_state_dict, strict=False)

            if missing_keys:
                print(f"❌ Missing keys: {missing_keys[:10]} ... ({len(missing_keys)} keys total)")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys[:10]} ... ({len(unexpected_keys)} keys total)")

        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")

class CLIPFineTuneDetector(nn.Module):
    """
    CLIP-based detector for binary classification with LoRA support.
    """
    def __init__(self, config: TrainingConfig):
        super().__init__()

        # Load configuration
        self.config = config
        self.device = self.config.device

        # Load CLIP backbone
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model

        # Classification head for binary classification
        self.head = nn.Linear(768, 2)

        # Load pretrained weights if specified in the config
        if self.config.model_weights:
            self._load_pretrained_weights(self.config.model_weights)

        # Apply LoRA if specified in the config
        if self.config.use_lora:
            self._inject_lora(self.config.lora_config)

    def forward(self, image_tensor):
        """
        Forward pass for fine-tuning.

        Args:
            image_tensor (torch.Tensor): Preprocessed input image tensor.

        Returns:
            torch.Tensor: Logits for each class.
        """
        features = self.backbone(image_tensor)['pooler_output']  # Ensure this key exists
        logits = self.head(features)
        return logits

    def _load_pretrained_weights(self, weights_path):
        """
        Loads pretrained weights and adjusts for potential name mismatches.

        Args:
            weights_path (str): Path to the pretrained weights.
        """
        state_dict = torch.load(weights_path, map_location=self.device)

        # Adjust for potential prefixes or mismatches
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load the state dictionary
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading weights: {unexpected_keys}")

    def _inject_lora(self, lora_config):
        """
        Injects LoRA adapters into the model's backbone.

        Args:
            lora_config (LoraConfig): Configuration for LoRA.
        """
        # Detect target modules for LoRA injection
        target_modules = [
            name for name, module in self.backbone.named_modules()
            if "k_proj" in name or "q_proj" in name or "v_proj" in name
        ]

        # Update the LoRA configuration with detected modules
        lora_config.target_modules = target_modules

        # Apply LoRA to the backbone
        self.backbone = get_peft_model(self.backbone, lora_config)


# if __name__ == "__main__":
#     # Initialize the training configuration
#     config = TrainingConfig()
#     config.use_lora = False
#     # Instantiate the model
#     model = CLIPFineTuneDetector(config=config).to(config.device)
#     print("Layers of CLIP Vision Model:")
#     for name, module in model.named_modules():
#         print(f"{name}: {module}")
#     # Create a dummy input tensor (batch size: 1, channels: 3, height: 224, width: 224)
#     dummy_input = torch.randn(1, 3, config.resolution, config.resolution).to(config.device)

#     # Test the forward pass
#     output = model(dummy_input)
#     print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    # Example usage
    weights_path = "/home/ginger/code/gderiddershanghai/deep-learning/weights/exception_DF40/xception.pth"

    # Instantiate the model
    model = XceptionDetector(num_classes=2, weights_path=weights_path).to("cuda" if torch.cuda.is_available() else "cpu")

    print("Model initialized successfully.")

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to("cuda" if torch.cuda.is_available() else "cpu")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")