import torch
from finetune_detectors import CLIPFineTuneDetector
from training_config import TrainingConfig

def get_finetune_detector(config: TrainingConfig):
    model = CLIPFineTuneDetector(config)
    model = model.to(config.device)
    return model

if __name__ == "__main__":
    from training_config import TrainingConfig
    config = TrainingConfig()
    detector = get_finetune_detector(config)
    dummy_input = torch.rand(1, 3, config.resolution, config.resolution).to(config.device)
    try:
        output = detector(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (1, 2), "Output shape mismatch!"
        print("Test passed: Output shape is correct.")
    except Exception as e:
        print(f"Test failed: {e}")