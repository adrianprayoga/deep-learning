import torch
from finetune_detectors import CLIPFineTuneDetector
from peft import LoraConfig

def get_finetune_detector(model_type, num_classes=2, apply_lora=False, weights_path=None, device='cpu'):
    """
    Returns a fine-tuning-specific detector model.
    """
    model = None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "clip":
        lora_config = None
        if apply_lora:
            lora_config = LoraConfig(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                r=8,  # Low-rank dimension
                lora_alpha=16,  # Scaling factor
                target_modules=["query", "key", "value"],  # Attention layers
                lora_dropout=0.1
            )
        model = CLIPFineTuneDetector(num_classes=num_classes, apply_lora=apply_lora, lora_config=lora_config)

        # Load pretrained weights
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    return model
