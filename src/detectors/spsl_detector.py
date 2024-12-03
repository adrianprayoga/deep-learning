'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the SPSLDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{liu2021spatial,
  title={Spatial-phase shallow learning: rethinking face forgery detection in frequency domain},
  author={Liu, Honggu and Li, Xiaodan and Zhou, Wenbo and Chen, Yuefeng and He, Yuan and Xue, Hui and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={772--781},
  year={2021}
}

@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}

Notes:
To ensure consistency in the comparison with other detectors, we have opted not to utilize the shallow Xception architecture. Instead, we are employing the original Xception model.
'''

import logging

import torch
import torch.nn as nn

# TODO: to add back
# from metrics.base_metrics_class import calculate_metrics_for_train

from networks import Xception

logger = logging.getLogger(__name__)
class SpslDetector(nn.Module):
    def __init__(self, config, load_weights=False):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config, load_weights)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config, load_weights):
        # prepare the backbone
        model_config = config['backbone_config']
        backbone = Xception(model_config)

        if load_weights:

        # To get a good performance, use the ImageNet-pretrained Xception model
        # pretrained here is path to saved weights

            print('loading trained weights from', config['pretrained'])
            if config['device'] == 'cpu':
                state_dict = torch.load(config['pretrained'], map_location=torch.device('cpu'))
            else:
                state_dict = torch.load(config['pretrained'])

            if any(key.startswith("module.backbone.") for key in state_dict.keys()):
                state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

            remove_first_layer = False
            if remove_first_layer:
                # remove conv1 from state_dict
                conv1_data = state_dict.pop('conv1.weight')
                missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)

                logger.info('Load pretrained model from {}'.format(config['pretrained']))
                # copy on conv1p
                # let new conv1 use old param to balance the network
                backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
                avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)  # average across the RGB channels
                # repeat the averaged weights across the 4 new channels
                backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)
            else:
                missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)

            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_func = CrossEntropyLoss()
        return loss_func

    def features(self, data_dict, phase_fea) -> torch.tensor:
        features = torch.cat((data_dict, phase_fea), dim=1)
        return self.backbone.features(features)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)


    def forward(self, data_dict, inference=False) -> dict:

        # get the phase features
        phase_fea = self.phase_without_amplitude(data_dict)
        # bp
        features = self.features(data_dict, phase_fea)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)
        # build the prediction dict for each output
        # prob takes the probability that it is label 1 --> fake/real?
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict['prob']

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

class CrossEntropyLoss():
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Computes the cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        # Compute the cross-entropy loss
        loss = self.loss_fn(inputs, targets)

        return loss