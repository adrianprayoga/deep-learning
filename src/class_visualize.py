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
model.load_state_dict(state_dict, strict=False)
class_names = ["Real", "Fake"]

# Initialize ClassVisualization
vis = ClassVisualization()

# Visualize "Fake" class
fake_image = vis.create_class_visualization(
    target_y=1,
    class_names=class_names,
    model=model,
    dtype=torch.float32,
    l2_reg=1e-3,
    learning_rate=25,
    num_iterations=1000,
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

