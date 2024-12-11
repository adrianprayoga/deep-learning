import torch
import torch.nn as nn
from transformers import CLIPModel
from torchvision import models
import timm

def get_detector(type, num_classes=2, load_weights=False, weights_path=None, device='cpu', config=None):
    model = None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if type == 'xception':
        model = XceptionDetector(num_classes=num_classes)
        if load_weights:
            if weights_path is None:
                print('Please check, something wrong as we are trying to load weights but path is None')
                return None

            missing_keys, unexpected_keys = None, None
            if weights_path is not None:
                # Load pre-trained weights
                state_dict = torch.load(weights_path, map_location=device)
                # Adjust for any prefixes in the state_dict keys (e.g., 'module.')
                if any(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                state_dict['backbone.fc.weight'] = state_dict.pop('backbone.last_linear.weight')
                state_dict['backbone.fc.bias'] = state_dict.pop('backbone.last_linear.bias')

                for key in list(state_dict.keys()):
                    if "adjust_channel" in key:
                        # print('removing', key, 'as it is unused')
                        state_dict.pop(key)

                # Load the state dictionary with relaxed strictness
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Optional: Log or handle missing/unexpected keys
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
    elif type == 'clip':
        model = CLIPDetector(num_classes=2)
        if load_weights:
            if weights_path is None:
                print('Please check, something wrong as we are trying to load weights but path is None')
                return None

            missing_keys, unexpected_keys = None, None
            if weights_path is not None:
                # Load pre-trained weights
                state_dict = torch.load(weights_path, map_location=device)
                # Adjust for any prefixes in the state_dict keys (e.g., 'module.')
                if any(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # Load the state dictionary with relaxed strictness
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Optional: Log or handle missing/unexpected keys
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
    elif type == 'spsl':
        model = SpslDetector(load_weights=True)
        if load_weights:
            if weights_path is None:
                print('Please check, something wrong as we are trying to load weights but path is None')
                return None

            missing_keys, unexpected_keys = None, None
            if weights_path is not None:
                # Load pre-trained weights
                state_dict = torch.load(weights_path, map_location=device)
                # Adjust for any prefixes in the state_dict keys (e.g., 'module.')
                if any(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                state_dict['backbone.fc.weight'] = state_dict.pop('backbone.last_linear.weight')
                state_dict['backbone.fc.bias'] = state_dict.pop('backbone.last_linear.bias')

                for key in list(state_dict.keys()):
                    if "adjust_channel" in key:
                        state_dict.pop(key)

                # Load the state dictionary with relaxed strictness
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

            # Optional: Log or handle missing/unexpected keys
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                print("all weights are fully loaded")

    return model

class CLIPDetector(nn.Module):
    """
    CLIP-based detector for binary classification (real vs fake).
    Designed for inference with CLIP Base (ViT-B/16).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Load the CLIP Base visual backbone
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
        
        # Classification head for binary classification
        self.head = nn.Linear(768, num_classes)  # 768 is the hidden size for ViT-B/16

    def forward(self, image_tensor):
        """
        Forward pass for inference.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Probabilities for each class (real and fake).
        """
        # Extract features using the backbone
        features = self.backbone(image_tensor)['pooler_output']

        # Classify features
        logits = self.head(features)

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

class XceptionDetector(nn.Module):
    """
    Xception-based detector for binary classification (real vs fake).
    Optimized for inference.
    """
    def __init__(self, num_classes=2, load_weights=False):
        super().__init__()
        # Load the pretrained Xception model
        self.backbone = timm.create_model('xception', pretrained=True)

        # Replace the classifier head for binary classification
        num_features = self.backbone.get_classifier().in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, image_tensor):
        """
        Forward pass for inference.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Probabilities for each class (real and fake).
        """
        # Pass through the model
        logits = self.backbone(image_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

class SpslDetector(nn.Module):

    # Code reference from
    # @article
    #
    # {yan2024df40,
    #  title = {DF40: Toward Next - Generation Deepfake Detection},
    # author = {Yan, Zhiyuan and Yao, Taiping and Chen, Shen and Zhao, Yandan and Fu, Xinghe and Zhu, Junwei and Luo,
    #           Donghao and Yuan, Li and Wang, Chengjie and Ding, Shouhong and others},
    # journal = {arXiv
    # preprint
    # arXiv: 2406.13495},
    # year = {2024}
    # }
    def __init__(self, load_weights=False):
        super().__init__()
        self.backbone = self.build_backbone()
        # self.loss_func = self.build_loss(config)

    def build_backbone(self, num_classes=2):
        # prepare the backbone
        # model_config = config['backbone_config']
        # backbone = Xception(model_config)
        backbone = timm.create_model('xception', pretrained=True)
        # Replace the classifier head for binary classification
        num_features = backbone.get_classifier().in_features
        backbone.fc = nn.Linear(num_features, num_classes)

        # if load_weights:
        #
        # # To get a good performance, use the ImageNet-pretrained Xception model
        # # pretrained here is path to saved weights
        #     if config['device'] == 'cpu':
        #         state_dict = torch.load(config['pretrained'], map_location=torch.device('cpu'))
        #     else:
        #         state_dict = torch.load(config['pretrained'])
        #
        #     if any(key.startswith("module.backbone.") for key in state_dict.keys()):
        #         state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
        #     state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        #
        #     remove_first_layer = False
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)

        #     else:
        #         missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
        #
        #     print("Missing keys:", missing_keys)
        #     print("Unexpected keys:", unexpected_keys)
        return backbone

    def forward(self, data_dict, inference=False):
        # get the phase features
        phase_fea = self.phase_without_amplitude(data_dict)
        # bp
        features = torch.cat((data_dict, phase_fea), dim=1)
        # get the prediction by classifier
        logits = self.backbone(features)
        # get the probability of the pred
        prob = torch.softmax(logits, dim=1)
        # build the prediction dict for each output
        # pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return prob

    def phase_without_amplitude(self, img):
        # Convert to grayscale
        # print(img)
        gray_img = torch.mean(img.to(torch.float32), dim=1, keepdim=True)  # shape: (batch_size, 1, 256, 256)
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img, dim=(-1, -2))
        # X = torch.fft.fftn(img)
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X, dim=(-1, -2)))
        # reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
        return reconstructed_x