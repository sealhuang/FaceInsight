# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


# VGG Face model
# input size: 224x224x3
VGG_Face = nn.Sequential(
        # block 1-1
        nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        # block 1-2
        nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        # block 2-1
        nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        # block 2-2
        nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        # block 2-3
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        # View
        Lambda(lambda x: x.view(x.size(0), -1)),
        # Linear -> 4096
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x),
                      nn.Linear(25088, 4096)),
        nn.ReLU(),
        nn.Dropout(0.5),
        # Linear -> 4096
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x),
                      nn.Linear(4096, 4096)),
        nn.ReLU(),
        nn.Dropout(0.5),
        # Linear -> 2622
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x),
                      nn.Linear(4096, 2622)),
)

