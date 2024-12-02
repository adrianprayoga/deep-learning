from torchvision import models
from pathlib import Path
import torch
import yaml
import os
from utils.model_loading import XceptionDetector
from utils.class_visualization import ClassVisualization
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
with open(os.path.join(PROJECT_DIR, 'src/config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Initialize model and class names
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = XceptionDetector(num_classes=2)
state_dict = torch.load(config["weights_path"], map_location=torch.device('cpu'))
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
state_dict['backbone.fc.weight'] = state_dict.pop('backbone.last_linear.weight')
state_dict['backbone.fc.bias'] = state_dict.pop('backbone.last_linear.bias')
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys:
    print(f"Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys: {unexpected_keys}")

# Initialize ClassVisualization
vis = ClassVisualization()

# Visualize "Fake" class
class_names = ["Real", "Fake"]
fake_image = vis.create_class_visualization(
    target_y=1,
    class_names=class_names,
    model=model,
    dtype=torch.float32,
    l2_reg=1e-3,
    learning_rate=25,
    num_iterations=100,
    blur_every=10,
    max_jitter=16,
    show_every=25,
    generate_plots=True
)

# Save or display the result
plt.imshow(fake_image)
plt.title("Fake Class Visualization")
plt.axis("off")
plt.show()

