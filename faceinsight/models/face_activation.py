# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function, division

import math
import torch
import torch.nn as nn


def l2_norm(input, axis=1):
    """L2-norm of the input tensor."""
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Arcface(nn.Module):
    """Arcface loss.
    Implementation of additive margin softmax loss in
    https://arxiv.org/abs/1801.05599.
    """
    def __init__(self, embedding_size=512, class_num=2, s=64., m=0.5):
        super(Arcface, self).__init__()
        self.class_num = class_num
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, class_num))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # the margin value, default is 0.5
        self.m = m 
        # scalar value default is 64
        # see normface https://arxiv.org/abs/1704.06369
        self.s = s 
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #output = torch.mm(embbedings, kernel_norm)
        # for numerical stability
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        # when theta not in [0,pi], use cosface instead
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        # a little bit hacky way to prevent in_place operation on cos_theta
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        # scale up in order to make softmax work, first introduced in normface
        output *= self.s
        return output


class Am_softmax(nn.Module):
    """Cosface loss.
    Implementation of additive margin softmax loss in
    https://arxiv.org/abs/1801.05599.
    """
    def __init__(self, embedding_size=512, class_num=2):
        super(Am_softmax, self).__init__()
        self.class_num = class_num
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # additive margin recommended by the paper
        self.m = 0.35
        # see normface https://arxiv.org/abs/1704.06369
        self.s = 30.

    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        # for numerical stability
        cos_theta = cos_theta.clamp(-1, 1)
        phi = cos_theta - self.m
        label = label.view(-1, 1) #size=(B, 1)
        index = cos_theta.data * 0.0 #size=(B, Classnum)
        index.scatter_(1, label.data.view(-1,1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        # only change the correct predicted output
        output[index] = phi[index]
        # scale up in order to make softmax work, first introduced in normface
        output *= self.s
        return output

