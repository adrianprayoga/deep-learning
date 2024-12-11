from torchvision import models
from pathlib import Path
import torch
import yaml
import os
from utils.config_utils import load_model_config
from utils.class_visualization import ClassVisualization
from utils.model_loading import get_detector
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
with open(os.path.join(PROJECT_DIR, 'src/config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Initialize model and class names
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = get_detector(config['model_type'],
                       config=load_model_config(config['model_type']),
                       load_weights=True,
                       weights_path=config["weights_path"])
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

