# import torch
# import numpy as np
# from captum.attr import GuidedGradCam
# import matplotlib.pyplot as plt
# import os

# from finetune_detectors import CLIPFineTuneDetector, TrainingConfig
# from inference_dataloader import InferenceDataset
# from captum_utils import compute_attributions, visualize_attr_maps


# def load_clip_model():
#     config = TrainingConfig()
#     config.use_lora = False
#     config.val_dataset1 = "/home/ginger/code/gderiddershanghai/deep-learning/data/face_finder_test"
#     model = CLIPFineTuneDetector(config=config).to(config.device)
#     return model, config.device


# def load_data(device, config):
#     dataset = InferenceDataset(config=config, use_val_set1=True)
#     images, labels, originals = [], [], []

#     for idx in range(len(dataset)):
#         X, y, _, original_image = dataset[idx]
#         images.append(X.to(device).unsqueeze(0))
#         labels.append(y)
#         originals.append(original_image)

#     return torch.cat(images, dim=0), torch.tensor(labels), np.concatenate(originals, axis=0)

# def run_gradcam(model, X_tensor, y_tensor, original_images, device, target_layer_name="backbone.encoder.layers[11].self_attn.out_proj"):
#     # Dynamically get the target layer
#     conv_module = eval(f"model.{target_layer_name}")
#     guided_grad_cam = GuidedGradCam(model, conv_module)

#     for i in range(10):
#         single_image = X_tensor[i:i+1].to(device).requires_grad_(True)
#         single_label = y_tensor[i:i+1].to(device)

#         # Compute Grad-CAM attributions
#         attr_ig = compute_attributions(guided_grad_cam, single_image, target=single_label)

#         # Ensure the attribution is not empty
#         attr_ig_np = attr_ig.squeeze().detach().cpu().numpy()
#         if attr_ig_np.ndim != 2 or attr_ig_np.size == 0:
#             print(f"[Warning] Empty attribution for image {i}. Skipping.")
#             continue

#         # Model prediction
#         pred = torch.argmax(model(single_image), dim=1)
#         correct_pred = torch.eq(single_label, pred).detach().cpu().numpy()

#         # Display visualization inline
#         plt.figure(figsize=(10, 8))
#         plt.title(f"Image {i} - CLIP GradCAM | Correct: {correct_pred[0]}")
#         plt.imshow(attr_ig_np, cmap="viridis", alpha=0.6)
#         plt.imshow(original_images[i], alpha=0.4)
#         plt.axis("off")
#         plt.show()

#         # Save visualization
#         OUTPUT_DIR = "src/finetuning/gradcam"
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         visualize_attr_maps(
#             os.path.join(OUTPUT_DIR, f"clip_gradcam_image_{i}.png"),
#             original_images[i:i+1], single_label.cpu(), ["real", "fake"], [attr_ig],
#             [f"Image {i} - CLIP GradCAM"], [correct_pred]
#         )



# if __name__ == '__main__':
#     possible_layers = [
#     # Final attention and MLP layers
#     "backbone.encoder.layers[11].self_attn.out_proj",
#     "backbone.encoder.layers[11].mlp.fc2",
    
#     # Mid-level layers for attention and MLP
#     "backbone.encoder.layers[6].self_attn.out_proj",
#     "backbone.encoder.layers[8].self_attn.out_proj",
#     "backbone.encoder.layers[9].mlp.fc2",
    
#     # Early layers for initial processing
#     "backbone.embeddings.patch_embedding",
#     "backbone.embeddings.position_embedding",

#     # Pre and post layer normalization
#     "backbone.pre_layrnorm",
#     "backbone.post_layernorm",

#     # Alternative attention layers
#     "backbone.encoder.layers[0].self_attn.out_proj",
#     "backbone.encoder.layers[4].self_attn.out_proj",
#     "backbone.encoder.layers[7].self_attn.out_proj",

#     # Classification head (if applicable)
#     "head"
# ]

#     # Load model and data
#     model, device = load_clip_model()
#     X_tensor, y_tensor, original_images = load_data(device, model.config)

#     # Enable gradient computation
#     X_tensor.requires_grad_(True)
#     print()
#     # Run Grad-CAM visualization
#     # possible_layers = []
#     for layer in possible_layers:
#         print(f'Testing Layer: {layer}')
#         run_gradcam(model, X_tensor, y_tensor, original_images, device, target_layer_name=layer)



import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Custom Hook for Capturing Gradients
class Hook:
    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self):
        return self.data
    
    @property
    def gradient(self):
        return self.data.grad

# Grad-CAM Implementation
def gradCAM(model, input_image, target, layer):
    input_image.requires_grad_(True)
    with Hook(layer) as hook:
        output = model(input_image)
        import torch.nn.functional as F

        # One-hot encode the target
        target_one_hot = F.one_hot(target, num_classes=output.shape[1]).float()

        # Compute backward pass
        output.backward(target_one_hot)


        grad = hook.gradient.float()
        act = hook.activation.float()

        # Average over the feature dimension (last)
        # Average over the feature dimension
        if grad.ndim == 3 and grad.shape[1] == 197:
            # Drop CLS token and average over features
            grad = grad[:, 1:, :]  # Shape: [1, 196, 768]
            act = act[:, 1:, :]  # Drop CLS token from activations too

            # Compute channel-wise importance
            alpha = grad.mean(dim=-1, keepdim=True)  # Shape: [1, 196, 1]

            # Weighted combination of activation maps
            gradcam = torch.sum(act * alpha, dim=-1)  # Shape: [1, 196]

            # Reshape to 14x14 grid
            gradcam = gradcam.reshape(1, 1, 14, 14)

            # Upscale to image size
            gradcam = F.interpolate(
                gradcam, size=(224, 224), mode='bicubic', align_corners=False
            ).squeeze()
        else:
            raise ValueError(f"Unexpected grad shape: {grad.shape}")




        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        gradcam = F.relu(gradcam)

    gradcam = F.interpolate(
        gradcam,
        input_image.shape[2:],
        mode="bicubic",
        align_corners=False
    )
    return gradcam

# Visualization Helper
def visualize_gradcam(image_np, attn_map, title="GradCAM", blur=False):
    def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-5)
    
    if blur:
        from scipy.ndimage import filters
        attn_map = filters.gaussian_filter(attn_map, sigma=0.02 * max(image_np.shape[:2]))

    attn_map = normalize(attn_map)
    cmap = plt.get_cmap("jet")(attn_map)[..., :3]

    result = (1 - attn_map[..., np.newaxis]) * image_np + attn_map[..., np.newaxis] * cmap

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.imshow(result)
    plt.axis("off")
    plt.show()

# Main Execution Block
if __name__ == "__main__":
    from finetune_detectors import CLIPFineTuneDetector, TrainingConfig
    from inference_dataloader import InferenceDataset

    # Load Model and Data
    config = TrainingConfig()
    model = CLIPFineTuneDetector(config=config).eval().to(config.device)
    dataset = InferenceDataset(config=config, use_val_set1=True)

    # Select Layer
    target_layer = model.backbone.encoder.layers[11].self_attn.out_proj

    # Process First Image from Dataset
    image_tensor, label, _, original_image = dataset[0]
    image_input = image_tensor.unsqueeze(0).to(config.device)
    label_tensor = torch.tensor([label]).to(config.device)

    # Run Grad-CAM
    attn_map = gradCAM(model, image_input, label_tensor, target_layer)
    attn_map_np = attn_map.squeeze().detach().cpu().numpy()

    # Visualize Result
    visualize_gradcam(original_image / 255.0, attn_map_np, title="CLIP Grad-CAM")
