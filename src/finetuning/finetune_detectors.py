# import torch
# from peft import get_peft_model, LoraConfig, TaskType
# from transformers import CLIPModel
# import torch.nn as nn
# from timm import create_model
# # https://github.com/huggingface/peft/issues/1988


# class CLIPFineTuneDetector(nn.Module):
#     def __init__(self, num_classes=2, apply_lora=False, lora_config=None, weights_path=None, device=None):
#         super().__init__()
        
#         self.device = device
#         self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
#         self.head = nn.Linear(768, num_classes)  # Classification head for binary classification

#         # Load pretrained weights if provided
#         if weights_path:
#             self._load_pretrained_weights(weights_path)

#         # Apply LoRA if specified
#         if apply_lora and lora_config is not None:
#             self._inject_lora(lora_config)

#     def forward(self, image_tensor):
#         # print("Before LoRA injection:")
#         # print(self.backbone.forward)

#         # self.backbone = get_peft_model(self.backbone, lora_config)

#         # print("After LoRA injection:")
#         # print(self.backbone.forward)

#         features = self.backbone(image_tensor)['pooler_output']  # Ensure this key exists
#         logits = self.head(features)
#         return logits

#     def _load_pretrained_weights(self, weights_path):
#         state_dict = torch.load(weights_path, map_location=self.device)
#         # Adjust for potential prefixes or name mismatches
#         if any(k.startswith("module.") for k in state_dict.keys()):
#             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#         missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
#         if missing_keys:
#             print(f"Missing keys when loading weights: {missing_keys}")
#         if unexpected_keys:
#             print(f"Unexpected keys when loading weights: {unexpected_keys}")

#     def _inject_lora(self, lora_config):
#         # print('------------------------------------------------------------')
#         # for name, module in self.backbone.named_modules():
#         #     print(name)
#         # print('------------------------------------------------------------')
#         target_modules = [
#             name for name, module in self.backbone.named_modules()
#             if "k_proj" in name or "q_proj" in name or "v_proj" in name
#         ]
#         # print(f"Matched target modules for LoRA injection: {target_modules}")
#         # print('------------------------------------------------------------')
#         lora_config.target_modules = target_modules
#         # print('********************************')
#         # print(list(self.backbone.named_modules()))
#         # print('********************************')
#         self.backbone = get_peft_model(self.backbone, lora_config)
#         # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
#         # print(list(self.backbone.named_modules()))
#         # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

# """
# Xception needs to be updated - didn't check after I created a training_config file
# """

# # class XceptionFineTuneDetector(nn.Module):
# #     """
# #     Xception-based detector for binary classification (real vs fake) with LoRA support.
# #     """
# #     def __init__(self, num_classes=2, apply_lora=False, lora_config=None, weights_path=None, device=None):
# #         super().__init__()

# #         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

# #         # Load the pretrained Xception model
# #         self.backbone = create_model("xception", pretrained=True)

# #         # Replace the classifier head for binary classification
# #         num_features = self.backbone.get_classifier().in_features
# #         self.backbone.fc = nn.Linear(num_features, num_classes)

# #         # Apply LoRA if specified
# #         if apply_lora and lora_config is not None:
# #             self._inject_lora(lora_config)

# #         # Load pretrained weights if provided
# #         if weights_path:
# #             self._load_pretrained_weights(weights_path)

# #     def forward(self, image_tensor):
# #         """
# #         Forward pass for fine-tuning.
# #         """
# #         logits = self.backbone(image_tensor)
# #         probabilities = torch.softmax(logits, dim=-1)
# #         return probabilities

# #     def _load_pretrained_weights(self, weights_path):
# #         """
# #         Loads and aligns pretrained weights to match the model's architecture.
# #         """
# #         state_dict = torch.load(weights_path, map_location=self.device)

# #         # Adjust for potential prefixes or name mismatches
# #         if any(k.startswith("module.") for k in state_dict.keys()):
# #             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# #         # Load the state dictionary into the model
# #         missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
# #         if missing_keys:
# #             print(f"Missing keys when loading weights: {missing_keys}")
# #         if unexpected_keys:
# #             print(f"Unexpected keys when loading weights: {unexpected_keys}")

# #     def _inject_lora(self, lora_config):
# #         """
# #         Injects LoRA adapters into the backbone based on target modules.
# #         """
# #         # Find convolutional modules for LoRA injection
# #         target_modules = [
# #             name for name, module in self.backbone.named_modules() if isinstance(module, nn.Conv2d)
# #         ]
# #         print(f"Matched target modules for LoRA injection: {target_modules}")

# #         # Update the LoRA config with the matched target modules
# #         lora_config.target_modules = target_modules

# #         # Inject LoRA into the backbone encoder layers
# #         self.backbone = get_peft_model(self.backbone, lora_config)



# if __name__ == "__main__":
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Configuration for LoRA (optional)
#     lora_config = LoraConfig(
#         # task_type=TaskType.FEATURE_EXTRACTION, 
#         r=8, 
#         lora_alpha=32, 
#         target_modules=None, 
#         lora_dropout=0.1, 
#         bias="none"
#     )

#     # Initialize the model
#     model = CLIPFineTuneDetector(
#         num_classes=2, 
#         apply_lora=True, 
#         lora_config=lora_config, 
#         device=device
#     )
#     model.to(device)

#     # Create a dummy input tensor (batch size: 1, channels: 3, height: 224, width: 224)
#     dummy_input = torch.randn(1, 3, 224, 224).to(device)

#     # Forward pass
#     try:
#         output = model(dummy_input)
#         print(f"Output shape: {output.shape}")
        
#         # Check if the output shape matches the expected dimensions
#         assert output.shape == (1, 2), "Output shape mismatch!"
#         print("Test passed: Output shape is correct.")
#     except Exception as e:
#         print(f"Test failed: {e}")
    
#     # lora_config = LoraConfig(
#     #     # task_type=TaskType.FEATURE_EXTRACTION,
#     #     r=8,
#     #     lora_alpha=16,
#     #     lora_dropout=0.1
#     # )
#     # print(TaskType)
#     # weights_path = "/home/ginger/code/gderiddershanghai/deep-learning/weights/exception_DF40/xception.pth"
#     # device = "cuda" if torch.cuda.is_available() else "cpu"

#     # # Without LoRA
#     # model_no_lora = XceptionFineTuneDetector(num_classes=2, weights_path=weights_path, device=device).to(device)

#     # # Test with dummy input
#     # dummy_input = torch.rand(1, 3, 224, 224).to(device)
#     # output_no_lora = model_no_lora(dummy_input)
#     # print(f"Output shape (no LoRA): {output_no_lora.shape}")

#     # # With LoRA
#     # model_with_lora = XceptionFineTuneDetector(
#     #     num_classes=2,
#     #     apply_lora=True,
#     #     lora_config=lora_config,
#     #     weights_path=weights_path,
#     #     device=device
#     # ).to(device)

#     # output_with_lora = model_with_lora(dummy_input)
#     # print(f"Output shape (with LoRA): {output_with_lora.shape}")
import torch
from peft import get_peft_model
from transformers import CLIPModel
import torch.nn as nn
from training_config import TrainingConfig  


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


if __name__ == "__main__":
    # Initialize the training configuration
    config = TrainingConfig()

    # Instantiate the model
    model = CLIPFineTuneDetector(config=config).to(config.device)

    # Create a dummy input tensor (batch size: 1, channels: 3, height: 224, width: 224)
    dummy_input = torch.randn(1, 3, config.resolution, config.resolution).to(config.device)

    # Test the forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
