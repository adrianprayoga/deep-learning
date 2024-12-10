# import torch
# from finetune_detectors import CLIPFineTuneDetector
# from peft import LoraConfig, TaskType

# def get_finetune_detector(model_type, 
#                           num_classes=2, 
#                           apply_lora=False, 
#                           weights_path=None, 
#                           device=None):
#     """
#     Returns a fine-tuning-specific detector model.
#     """
#     model = None
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     if model_type == "clip":
#         lora_config = None
#         if apply_lora:
#             lora_config = LoraConfig(
#                 task_type=TaskType.FEATURE_EXTRACTION,
#                 r=8,  # Low-rank dimension
#                 lora_alpha=16,  # Scaling factor
#                 target_modules=["query", "key", "value"],  # Attention layers
#                 lora_dropout=0.1
#             )
#         model = CLIPFineTuneDetector(num_classes=num_classes, 
#                                      apply_lora=apply_lora, 
#                                      lora_config=lora_config)

#         # Load pretrained weights
#         if weights_path is not None:
#             state_dict = torch.load(weights_path, map_location=device)
#             model.load_state_dict(state_dict, strict=False)

#     model = model.to(device)
#     return model


# if __name__ == "__main__":
#     import torch
#     from peft import TaskType, LoraConfig
#     from transformers import CLIPModel
#     from finetune_detectors import CLIPFineTuneDetector

#     # Path to pretrained weights
#     WEIGHT_PATH = "/home/ginger/code/gderiddershanghai/deep-learning/weights/clip_EFS/clip.pth"

#     # Set device
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Step 1: Load the vision model from CLIP
#     print("Detecting and validating LoRA target modules...")
#     vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model

#     # Step 2: List all state dict keys for debugging
#     state_dict_keys = list(vision_model.state_dict().keys())
#     print("State dict keys (sample):", state_dict_keys[:20])

#     # Step 3: Dynamically detect target modules for LoRA injection
#     target_modules = [
#         name for name, module in vision_model.named_modules()
#         if "k_proj" in name or "q_proj" in name or "v_proj" in name
#     ]
#     print("\n--------------------------------------------------")
#     print("Matched target modules for LoRA injection (sample):", target_modules[:10])

#     # Step 4: Validate that detected target modules match state dictionary
#     state_dict_modules = set([key.rsplit('.', 1)[0] for key in state_dict_keys])
#     unmatched = [module for module in target_modules if module not in state_dict_modules]

#     if unmatched:
#         print(f"Unmatched target modules: {unmatched}")
#     else:
#         print("All target modules match the state dict.")

#     filtered_target_modules = []
#     for module_name in target_modules:
#         module = dict(vision_model.named_modules()).get(module_name)
#         if isinstance(module, torch.nn.Linear):  # Example: Only inject into nn.Linear layers
#             filtered_target_modules.append(module_name)

#     print("Filtered target modules for LoRA injection:", filtered_target_modules)

#     print('-------------------------------------------------------------------')

#     # Optional: Proceed with LoRA configuration and testing
#     lora_config = LoraConfig(
#         r=8,  # Low-rank adaptation dimension
#         lora_alpha=16,  # Scaling factor
#         target_modules=target_modules,
#         lora_dropout=0.1,
#     )

#     print("\nTesting CLIPFineTuneDetector with LoRA...")

#     # Step 5: Initialize CLIPFineTuneDetector with LoRA
#     detector_with_lora = CLIPFineTuneDetector(
#         num_classes=2,
#         apply_lora=True,
#         lora_config=lora_config,
#     ).to(device)

#     # Step 6: Test inference with a dummy input
#     dummy_input = torch.rand(1, 3, 224, 224).to(device)
#     output_with_lora = detector_with_lora(dummy_input)
#     print(f"Output shape (with LoRA): {output_with_lora.shape}")

import torch
from finetune_detectors import CLIPFineTuneDetector
from training_config import TrainingConfig


def get_finetune_detector(config: TrainingConfig):
    """
    Returns a fine-tuning-specific detector model using the given configuration.

    Args:
        config (TrainingConfig): Configuration object containing model settings.

    Returns:
        nn.Module: The fine-tuned detector model.
    """
    # Initialize the model using the configuration
    model = CLIPFineTuneDetector(config)

    # Move the model to the specified device
    model = model.to(config.device)
    return model


if __name__ == "__main__":
    # Load the training configuration
    from training_config import TrainingConfig

    # Initialize the configuration
    config = TrainingConfig()

    # Instantiate the fine-tuning detector
    print("Detecting and validating LoRA target modules...")
    detector = get_finetune_detector(config)

    # Create a dummy input tensor (batch size: 1, channels: 3, height: 224, width: 224)
    dummy_input = torch.rand(1, 3, config.resolution, config.resolution).to(config.device)

    # Test the forward pass
    try:
        output = detector(dummy_input)
        print(f"Output shape: {output.shape}")

        # Verify the output dimensions
        assert output.shape == (1, 2), "Output shape mismatch!"
        print("Test passed: Output shape is correct.")
    except Exception as e:
        print(f"Test failed: {e}")
