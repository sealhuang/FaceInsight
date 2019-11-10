# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import torch
import torchvision.transforms as transforms

from faceinsight.models.shufflenet_v2 import ShuffleNetV2

#from config import configurations


class Recognizer():
    """Face Recognizer based on ShuffleNet."""
    
    def __init__(self, device='cpu'):
        """Model initialization."""
        self.device = device

        # image preprocessing config
        RGB_MEAN = [0.5, 0.5, 0.5]
        RGB_STD = [0.5, 0.5, 0.5]
        self.img_transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=RGB_MEAN,
                                                         std=RGB_STD),
                            ])

        # load model
        model_file = 'data/Backbone_shufflenet_v2_x1_0_Epoch_40_checkpoint.pth'
        self.model = ShuffleNetV2(n_class=512, input_size=224, width_mult=1.0)
        self.model.load_state_dict(torch.load(model_file,
                                map_location=lambda storage, loc: storage))
        if device=='gpu':
            self.model = self.model.cuda()
        self.model.eval()

    def infer(self, imgs):
        """Input:
            `imgs`: list of PIL.Image.
        
        Return:
            `face_features`: numpy.array of shape [n_faces, embedding_size]
        """
        # image transformation
        inputs = []
        for sample in imgs:
            inputs.append(self.img_transform(sample))
        inputs = torch.stack(inputs)

        # compute output
        inputs = inputs.to(self.device)
        features = self.model(inputs)
        #print(features.cpu().data.numpy().shape)

        return features.cpu().data.numpy()

