import os
from pathlib import Path

import torch
import yaml
from captum.attr import Saliency, GuidedGradCam

from visualizers.image_utils import preprocess
from utils.dataloader import InferenceDataset
from utils.model_loading import get_detector
from visualizers.captum_utils import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

model_types = ['spsl', 'xception', 'clip']
ds_types = ['fs', 'fr', 'efs']


def get_all_models(config):
    models = {}
    for model_type in model_types:
        models[model_type] = {}
        for train in ds_types:
            weight_conf = model_type +"_" + train + "_path"
            model = get_detector(model_type,
                                 load_weights=True,
                                 weights_path=config[weight_conf])
            for param in model.parameters():
                param.requires_grad = False
            models[model_type][train] = model

    return models

if __name__ == '__main__':

    ## FILE THAT NEEDS TO BE UPDATED
    # Change the file below to your local config name
    # run python inference.py from main folder e.g. python src/inference.py

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    with open(os.path.join(PROJECT_DIR, 'src/config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    # model = get_detector(config['model_type'],
    #                         config=load_model_config(config['model_type']),
    #                         load_weights=True,
    #                         weights_path=config["weights_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.join('visualization', 'gradcam'), exist_ok=True)
    os.makedirs(os.path.join('visualization', 'saliency'), exist_ok=True)

    # We don't want to train the model, so tell PyTorch not to compute gradients
    # with respect to model parameters.


    dataset_eval = InferenceDataset(root_dir=config['viz_data_root_dir'],
                                    resolution=224,
                                    model_name=config['model_type'])

    # I am not sure why the batch only load 1 at a time
    all_images = []
    all_labels = []
    all_image_paths = []
    all_original_images = []

    for batch in dataset_eval:
        X, labels, image_paths, original_image = batch
        images = X.to(device)
        if len(X.shape) == 3:  # If single image without batch dim
            X = X.unsqueeze(0)  # Add batch dimension
            original_image = original_image[np.newaxis, :]
        all_images.append(X)
        all_labels.append(labels)
        all_image_paths.extend(image_paths)
        all_original_images.append(original_image)

    # SALIENCY
    all_images = torch.cat(all_images, dim=0)  # Combine all tensors into one batch
    y_tensor = torch.tensor(all_labels)  # Convert labels into a single tensor
    all_original_images = np.concatenate(all_original_images, axis=0)  # Combine into one array
    class_names = ['real', 'fake']
    X_tensor = all_images.requires_grad_(True)

    all_models = get_all_models(config)

    # SALIENCY
    for model_type in model_types:
        attributes = []
        titles = []
        preds = []

        for train in ds_types:
            model = all_models[model_type][train]
            argmax = torch.argmax(model(X_tensor), dim=1)
            correct_prediction = torch.eq(y_tensor, argmax)
            preds.append(correct_prediction.detach().cpu().numpy())

            # print(y_tensor)
            # print(torch.eq(y_tensor, argmax))
            # print('prediction', model_type, train, argmax)

            saliency = Saliency(model)
            attr_ig = compute_attributions(saliency, X_tensor, target=y_tensor)
            titles.append(model_type + "_" + train)
            attributes.append(attr_ig)

            visualize_attr_maps('visualization/saliency/'+model_type+'.png', all_original_images, y_tensor, class_names,
                                attributes, titles, preds)

    # GRADCAM
    for model_type in model_types:
        if model_type == 'clip':
            continue

        attributes = []
        titles = []
        preds = []

        for train in ds_types:
            model = all_models[model_type][train]
            argmax = torch.argmax(model(X_tensor), dim=1)
            correct_prediction = torch.eq(y_tensor, argmax)
            preds.append(correct_prediction.detach().cpu().numpy())
            conv_module = model.backbone.conv4
            # conv_module = model.backbone.encoder.layers[11].self_attn.out_proj

            guidedGramCam = GuidedGradCam(model, conv_module)
            attr_ig = compute_attributions(guidedGramCam, X_tensor, target=y_tensor)
            attributes.append(attr_ig)
            titles.append(model_type + "_" + train)

            # print('attr_ig', attr_ig.shape)

            visualize_attr_maps('visualization/gradcam/'+model_type+'.png', all_original_images, y_tensor, class_names,
                                attributes, titles, preds)





