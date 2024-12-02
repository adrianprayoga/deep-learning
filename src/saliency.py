import os
from pathlib import Path

import torch
import yaml
from captum.attr import Saliency

from visualizers.image_utils import preprocess
from utils.dataloader import InferenceDataset
from utils.model_loading import get_detector
from visualizers.captum_utils import *
from utils.config_utils import load_model_config

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if __name__ == '__main__':

    ## FILE THAT NEEDS TO BE UPDATED
    # Change the file below to your local config name
    # run python inference.py from main folder e.g. python src/inference.py

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    with open(os.path.join(PROJECT_DIR, 'src/config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    model = get_detector(config['model_type'],
                            config=load_model_config(config['model_type']),
                            load_weights=True,
                            weights_path=config["weights_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # We don't want to train the model, so tell PyTorch not to compute gradients
    # with respect to model parameters.
    for param in model.parameters():
        param.requires_grad = False

    ## load image, tensor, tensor, class_names: string
    ## maybe we provide specific
    # X, y, class_names = load_imagenet_val(num=5)
    dataset_eval = InferenceDataset(root_dir=config['saliency_data_root_dir'],
                                    resolution=224,
                                    model_name=config['model_type'])

    # I am not sure why the batch only load 1 at a time
    all_images = []
    all_labels = []
    all_image_paths = []
    all_original_images = []

    for batch in dataset_eval:

        X, labels, image_paths, original_image = batch

        print(X.shape, labels, image_paths, original_image.shape)

        images = X.to(device)


        if len(X.shape) == 3:  # If single image without batch dim
            X = X.unsqueeze(0)  # Add batch dimension
            original_image = original_image[np.newaxis, :]

            # Store the batch data
        all_images.append(X)
        all_labels.append(labels)
        all_image_paths.extend(image_paths)
        all_original_images.append(original_image)

        # if len(images.shape) == 3:  # If single image without batch dim
        #     images = images.unsqueeze(0)  # Add batch dimension
        #     X = X.unsqueeze(0)
        #     original_image = original_image[np.newaxis, :]

    all_images = torch.cat(all_images, dim=0)  # Combine all tensors into one batch
    y_tensor = torch.tensor(all_labels)  # Convert labels into a single tensor
    all_original_images = np.concatenate(all_original_images, axis=0)  # Combine into one array
    class_names = ['real', 'fake']

    X_tensor = all_images.requires_grad_(True)
    saliency = Saliency(model)
    attr_ig = compute_attributions(saliency, X_tensor, target=y_tensor)
    visualize_attr_maps('visualization/saliency/captum.png', all_original_images, y_tensor, class_names, [attr_ig], ['Saliency'])


