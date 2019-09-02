# vi: set ft=python sts=4 ts=4 sw=4 et:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from faceinsight.models.shufflefacenet import ShuffleNetV2


class ShuffleFaceNetClfier(nn.Module):
    def __init__(self, class_num):
        super(ShuffleFaceNetClfier, self).__init__()
        self.fc1 = nn.Linear(512, class_num, bias=False)

    def forward(self, x):
        
        x = self.fc1(x)
        return x

class ShuffleFaceNet(nn.Module):
    def __init__(self, backbone_file, clfier_file, device):
        super(ShuffleFaceNet, self).__init__()
        self.backbone = ShuffleNetV2(n_class=512, input_size=224,
                                     width_mult=1.0)
        self.classifier = ShuffleFaceNetClfier(2)
        # load model weights
        self.backbone.load_state_dict(torch.load(backbone_file,
                                     map_location=lambda storage, loc: storage))
        self.classifier.load_state_dict(torch.load(clfier_file,
                                     map_location=lambda storage, loc: storage))
        if device=='gpu':
            self.backbone = self.backbone.cuda()
            self.classifier = self.classifier.cuda()
        self.backbone.eval()
        self.classifier.eval()

    def forward(self, x):
        feat = self.backbone(x)
        x = F.softmax(self.classifier(feat), dim=1)
        return x
 

def img_preprocess(image):
    """Image preprocess pipeline."""
    # define transforms
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    test_transform = transforms.Compose([transforms.Resize(250),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    img = test_transform(image)
    return img.unsqueeze(0)

