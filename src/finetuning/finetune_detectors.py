# # https://github.com/huggingface/peft/issues/1988
import torch
from peft import get_peft_model
from transformers import CLIPModel
import torch.nn as nn
from training_config import TrainingConfig  
import timm

class XceptionDetector(nn.Module):
    def __init__(self, num_classes=2, weights_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.backbone = timm.create_model("xception", pretrained=False)
        num_features = self.backbone.get_classifier().in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        if weights_path:
            self._load_pretrained_weights(weights_path)

    def forward(self, image_tensor):
        logits = self.backbone(image_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def _load_pretrained_weights(self, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            corrected_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "").replace("backbone.", "")
                if new_key == "last_linear.weight":
                    new_key = "fc.weight"
                elif new_key == "last_linear.bias":
                    new_key = "fc.bias"
                corrected_state_dict[new_key] = v
            missing_keys, unexpected_keys = self.backbone.load_state_dict(corrected_state_dict, strict=False)
            if missing_keys:
                print(f" Missing keys: {missing_keys[:10]} ... ({len(missing_keys)} keys total)")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:10]} ... ({len(unexpected_keys)} keys total)")
        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")

class CLIPFineTuneDetector(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
        self.head = nn.Linear(768, 2)
        if self.config.model_weights:
            self._load_pretrained_weights(self.config.model_weights)
        if self.config.use_lora:
            self._inject_lora(self.config.lora_config)

    def forward(self, image_tensor):
        features = self.backbone(image_tensor)['pooler_output']
        logits = self.head(features)
        return logits

    def _load_pretrained_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location=self.device)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading weights: {unexpected_keys}")

    def _inject_lora(self, lora_config):
        target_modules = [
            name for name, module in self.backbone.named_modules()
            if "k_proj" in name or "q_proj" in name or "v_proj" in name
        ]
        lora_config.target_modules = target_modules
        self.backbone = get_peft_model(self.backbone, lora_config)

if __name__ == "__main__":
    weights_path = "/home/ginger/code/gderiddershanghai/deep-learning/weights/exception_DF40/xception.pth"
    model = XceptionDetector(num_classes=2, weights_path=weights_path).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model initialized successfully.")
    dummy_input = torch.randn(1, 3, 224, 224).to("cuda" if torch.cuda.is_available() else "cpu")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")