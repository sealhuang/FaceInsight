import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv1_7x7_s2 = self.__conv(2, name='conv1/7x7_s2', in_channels=3, out_channels=64, kernel_size=(7L, 7L), stride=(2L, 2L), groups=1, bias=False)
        self.conv1_7x7_s2_bn = self.__batch_normalization(2, 'conv1/7x7_s2/bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_1_1x1_reduce = self.__conv(2, name='conv2_1_1x1_reduce', in_channels=64, out_channels=64, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_1_1x1_proj = self.__conv(2, name='conv2_1_1x1_proj', in_channels=64, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_1_1x1_reduce_bn = self.__batch_normalization(2, 'conv2_1_1x1_reduce/bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_1_1x1_proj_bn = self.__batch_normalization(2, 'conv2_1_1x1_proj/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_1_3x3 = self.__conv(2, name='conv2_1_3x3', in_channels=64, out_channels=64, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_1_3x3_bn = self.__batch_normalization(2, 'conv2_1_3x3/bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_1_1x1_increase = self.__conv(2, name='conv2_1_1x1_increase', in_channels=64, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_1_1x1_increase_bn = self.__batch_normalization(2, 'conv2_1_1x1_increase/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_2_1x1_reduce = self.__conv(2, name='conv2_2_1x1_reduce', in_channels=256, out_channels=64, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_2_1x1_reduce_bn = self.__batch_normalization(2, 'conv2_2_1x1_reduce/bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_2_3x3 = self.__conv(2, name='conv2_2_3x3', in_channels=64, out_channels=64, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_2_3x3_bn = self.__batch_normalization(2, 'conv2_2_3x3/bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_2_1x1_increase = self.__conv(2, name='conv2_2_1x1_increase', in_channels=64, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_2_1x1_increase_bn = self.__batch_normalization(2, 'conv2_2_1x1_increase/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_3_1x1_reduce = self.__conv(2, name='conv2_3_1x1_reduce', in_channels=256, out_channels=64, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_3_1x1_reduce_bn = self.__batch_normalization(2, 'conv2_3_1x1_reduce/bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_3_3x3 = self.__conv(2, name='conv2_3_3x3', in_channels=64, out_channels=64, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_3_3x3_bn = self.__batch_normalization(2, 'conv2_3_3x3/bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.conv2_3_1x1_increase = self.__conv(2, name='conv2_3_1x1_increase', in_channels=64, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv2_3_1x1_increase_bn = self.__batch_normalization(2, 'conv2_3_1x1_increase/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_1_1x1_proj = self.__conv(2, name='conv3_1_1x1_proj', in_channels=256, out_channels=512, kernel_size=(1L, 1L), stride=(2L, 2L), groups=1, bias=False)
        self.conv3_1_1x1_reduce = self.__conv(2, name='conv3_1_1x1_reduce', in_channels=256, out_channels=128, kernel_size=(1L, 1L), stride=(2L, 2L), groups=1, bias=False)
        self.conv3_1_1x1_proj_bn = self.__batch_normalization(2, 'conv3_1_1x1_proj/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_1_1x1_reduce_bn = self.__batch_normalization(2, 'conv3_1_1x1_reduce/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_1_3x3 = self.__conv(2, name='conv3_1_3x3', in_channels=128, out_channels=128, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_1_3x3_bn = self.__batch_normalization(2, 'conv3_1_3x3/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_1_1x1_increase = self.__conv(2, name='conv3_1_1x1_increase', in_channels=128, out_channels=512, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_1_1x1_increase_bn = self.__batch_normalization(2, 'conv3_1_1x1_increase/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_2_1x1_reduce = self.__conv(2, name='conv3_2_1x1_reduce', in_channels=512, out_channels=128, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_2_1x1_reduce_bn = self.__batch_normalization(2, 'conv3_2_1x1_reduce/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_2_3x3 = self.__conv(2, name='conv3_2_3x3', in_channels=128, out_channels=128, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_2_3x3_bn = self.__batch_normalization(2, 'conv3_2_3x3/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_2_1x1_increase = self.__conv(2, name='conv3_2_1x1_increase', in_channels=128, out_channels=512, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_2_1x1_increase_bn = self.__batch_normalization(2, 'conv3_2_1x1_increase/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_3_1x1_reduce = self.__conv(2, name='conv3_3_1x1_reduce', in_channels=512, out_channels=128, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_3_1x1_reduce_bn = self.__batch_normalization(2, 'conv3_3_1x1_reduce/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_3_3x3 = self.__conv(2, name='conv3_3_3x3', in_channels=128, out_channels=128, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_3_3x3_bn = self.__batch_normalization(2, 'conv3_3_3x3/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_3_1x1_increase = self.__conv(2, name='conv3_3_1x1_increase', in_channels=128, out_channels=512, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_3_1x1_increase_bn = self.__batch_normalization(2, 'conv3_3_1x1_increase/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_4_1x1_reduce = self.__conv(2, name='conv3_4_1x1_reduce', in_channels=512, out_channels=128, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_4_1x1_reduce_bn = self.__batch_normalization(2, 'conv3_4_1x1_reduce/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_4_3x3 = self.__conv(2, name='conv3_4_3x3', in_channels=128, out_channels=128, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_4_3x3_bn = self.__batch_normalization(2, 'conv3_4_3x3/bn', num_features=128, eps=9.99999974738e-06, momentum=0.0)
        self.conv3_4_1x1_increase = self.__conv(2, name='conv3_4_1x1_increase', in_channels=128, out_channels=512, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv3_4_1x1_increase_bn = self.__batch_normalization(2, 'conv3_4_1x1_increase/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_1_1x1_proj = self.__conv(2, name='conv4_1_1x1_proj', in_channels=512, out_channels=1024, kernel_size=(1L, 1L), stride=(2L, 2L), groups=1, bias=False)
        self.conv4_1_1x1_reduce = self.__conv(2, name='conv4_1_1x1_reduce', in_channels=512, out_channels=256, kernel_size=(1L, 1L), stride=(2L, 2L), groups=1, bias=False)
        self.conv4_1_1x1_proj_bn = self.__batch_normalization(2, 'conv4_1_1x1_proj/bn', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_1_1x1_reduce_bn = self.__batch_normalization(2, 'conv4_1_1x1_reduce/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_1_3x3 = self.__conv(2, name='conv4_1_3x3', in_channels=256, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_1_3x3_bn = self.__batch_normalization(2, 'conv4_1_3x3/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_1_1x1_increase = self.__conv(2, name='conv4_1_1x1_increase', in_channels=256, out_channels=1024, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_1_1x1_increase_bn = self.__batch_normalization(2, 'conv4_1_1x1_increase/bn', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_2_1x1_reduce = self.__conv(2, name='conv4_2_1x1_reduce', in_channels=1024, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_2_1x1_reduce_bn = self.__batch_normalization(2, 'conv4_2_1x1_reduce/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_2_3x3 = self.__conv(2, name='conv4_2_3x3', in_channels=256, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_2_3x3_bn = self.__batch_normalization(2, 'conv4_2_3x3/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_2_1x1_increase = self.__conv(2, name='conv4_2_1x1_increase', in_channels=256, out_channels=1024, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_2_1x1_increase_bn = self.__batch_normalization(2, 'conv4_2_1x1_increase/bn', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_3_1x1_reduce = self.__conv(2, name='conv4_3_1x1_reduce', in_channels=1024, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_3_1x1_reduce_bn = self.__batch_normalization(2, 'conv4_3_1x1_reduce/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_3_3x3 = self.__conv(2, name='conv4_3_3x3', in_channels=256, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_3_3x3_bn = self.__batch_normalization(2, 'conv4_3_3x3/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_3_1x1_increase = self.__conv(2, name='conv4_3_1x1_increase', in_channels=256, out_channels=1024, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_3_1x1_increase_bn = self.__batch_normalization(2, 'conv4_3_1x1_increase/bn', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_4_1x1_reduce = self.__conv(2, name='conv4_4_1x1_reduce', in_channels=1024, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_4_1x1_reduce_bn = self.__batch_normalization(2, 'conv4_4_1x1_reduce/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_4_3x3 = self.__conv(2, name='conv4_4_3x3', in_channels=256, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_4_3x3_bn = self.__batch_normalization(2, 'conv4_4_3x3/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_4_1x1_increase = self.__conv(2, name='conv4_4_1x1_increase', in_channels=256, out_channels=1024, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_4_1x1_increase_bn = self.__batch_normalization(2, 'conv4_4_1x1_increase/bn', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_5_1x1_reduce = self.__conv(2, name='conv4_5_1x1_reduce', in_channels=1024, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_5_1x1_reduce_bn = self.__batch_normalization(2, 'conv4_5_1x1_reduce/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_5_3x3 = self.__conv(2, name='conv4_5_3x3', in_channels=256, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_5_3x3_bn = self.__batch_normalization(2, 'conv4_5_3x3/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_5_1x1_increase = self.__conv(2, name='conv4_5_1x1_increase', in_channels=256, out_channels=1024, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_5_1x1_increase_bn = self.__batch_normalization(2, 'conv4_5_1x1_increase/bn', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_6_1x1_reduce = self.__conv(2, name='conv4_6_1x1_reduce', in_channels=1024, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_6_1x1_reduce_bn = self.__batch_normalization(2, 'conv4_6_1x1_reduce/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_6_3x3 = self.__conv(2, name='conv4_6_3x3', in_channels=256, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_6_3x3_bn = self.__batch_normalization(2, 'conv4_6_3x3/bn', num_features=256, eps=9.99999974738e-06, momentum=0.0)
        self.conv4_6_1x1_increase = self.__conv(2, name='conv4_6_1x1_increase', in_channels=256, out_channels=1024, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv4_6_1x1_increase_bn = self.__batch_normalization(2, 'conv4_6_1x1_increase/bn', num_features=1024, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_1_1x1_proj = self.__conv(2, name='conv5_1_1x1_proj', in_channels=1024, out_channels=2048, kernel_size=(1L, 1L), stride=(2L, 2L), groups=1, bias=False)
        self.conv5_1_1x1_reduce = self.__conv(2, name='conv5_1_1x1_reduce', in_channels=1024, out_channels=512, kernel_size=(1L, 1L), stride=(2L, 2L), groups=1, bias=False)
        self.conv5_1_1x1_proj_bn = self.__batch_normalization(2, 'conv5_1_1x1_proj/bn', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_1_1x1_reduce_bn = self.__batch_normalization(2, 'conv5_1_1x1_reduce/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_1_3x3 = self.__conv(2, name='conv5_1_3x3', in_channels=512, out_channels=512, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_1_3x3_bn = self.__batch_normalization(2, 'conv5_1_3x3/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_1_1x1_increase = self.__conv(2, name='conv5_1_1x1_increase', in_channels=512, out_channels=2048, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_1_1x1_increase_bn = self.__batch_normalization(2, 'conv5_1_1x1_increase/bn', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_2_1x1_reduce = self.__conv(2, name='conv5_2_1x1_reduce', in_channels=2048, out_channels=512, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_2_1x1_reduce_bn = self.__batch_normalization(2, 'conv5_2_1x1_reduce/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_2_3x3 = self.__conv(2, name='conv5_2_3x3', in_channels=512, out_channels=512, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_2_3x3_bn = self.__batch_normalization(2, 'conv5_2_3x3/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_2_1x1_increase = self.__conv(2, name='conv5_2_1x1_increase', in_channels=512, out_channels=2048, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_2_1x1_increase_bn = self.__batch_normalization(2, 'conv5_2_1x1_increase/bn', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_3_1x1_reduce = self.__conv(2, name='conv5_3_1x1_reduce', in_channels=2048, out_channels=512, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_3_1x1_reduce_bn = self.__batch_normalization(2, 'conv5_3_1x1_reduce/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_3_3x3 = self.__conv(2, name='conv5_3_3x3', in_channels=512, out_channels=512, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_3_3x3_bn = self.__batch_normalization(2, 'conv5_3_3x3/bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.conv5_3_1x1_increase = self.__conv(2, name='conv5_3_1x1_increase', in_channels=512, out_channels=2048, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=False)
        self.conv5_3_1x1_increase_bn = self.__batch_normalization(2, 'conv5_3_1x1_increase/bn', num_features=2048, eps=9.99999974738e-06, momentum=0.0)
        self.feat_extract_1 = self.__dense(name = 'feat_extract_1', in_features = 2048, out_features = 256, bias = True)
        self.classifier_8631_f_1 = self.__dense(name = 'classifier_8631_f_1', in_features = 256, out_features = 8631, bias = True)

    def forward(self, x):
        conv1_7x7_s2_pad = F.pad(x, (3L, 3L, 3L, 3L))
        conv1_7x7_s2    = self.conv1_7x7_s2(conv1_7x7_s2_pad)
        conv1_7x7_s2_bn = self.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_relu_7x7_s2 = F.relu(conv1_7x7_s2_bn)
        pool1_3x3_s2_pad = F.pad(conv1_relu_7x7_s2, (0L, 1L, 0L, 1L), value=float('-inf'))
        pool1_3x3_s2    = F.max_pool2d(pool1_3x3_s2_pad, kernel_size=(3L, 3L), stride=(2L, 2L), padding=0, ceil_mode=False)
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1_1x1_reduce_relu = F.relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3_pad = F.pad(conv2_1_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv2_1_3x3     = self.conv2_1_3x3(conv2_1_3x3_pad)
        conv2_1_3x3_bn  = self.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_relu = F.relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3_relu)
        conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1         = conv2_1_1x1_increase_bn + conv2_1_1x1_proj_bn
        conv2_1_relu    = F.relu(conv2_1)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1_relu)
        conv2_2_1x1_reduce_bn = self.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_relu = F.relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3_pad = F.pad(conv2_2_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv2_2_3x3     = self.conv2_2_3x3(conv2_2_3x3_pad)
        conv2_2_3x3_bn  = self.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_relu = F.relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3_relu)
        conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2         = conv2_1_relu + conv2_2_1x1_increase_bn
        conv2_2_relu    = F.relu(conv2_2)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2_relu)
        conv2_3_1x1_reduce_bn = self.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_relu = F.relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3_pad = F.pad(conv2_3_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv2_3_3x3     = self.conv2_3_3x3(conv2_3_3x3_pad)
        conv2_3_3x3_bn  = self.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_relu = F.relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3_relu)
        conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3         = conv2_2_relu + conv2_3_1x1_increase_bn
        conv2_3_relu    = F.relu(conv2_3)
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3_relu)
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3_relu)
        conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1_1x1_reduce_bn = self.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_relu = F.relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3_pad = F.pad(conv3_1_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv3_1_3x3     = self.conv3_1_3x3(conv3_1_3x3_pad)
        conv3_1_3x3_bn  = self.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_relu = F.relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3_relu)
        conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1         = conv3_1_1x1_increase_bn + conv3_1_1x1_proj_bn
        conv3_1_relu    = F.relu(conv3_1)
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1_relu)
        conv3_2_1x1_reduce_bn = self.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_relu = F.relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3_pad = F.pad(conv3_2_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv3_2_3x3     = self.conv3_2_3x3(conv3_2_3x3_pad)
        conv3_2_3x3_bn  = self.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_relu = F.relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3_relu)
        conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2         = conv3_1_relu + conv3_2_1x1_increase_bn
        conv3_2_relu    = F.relu(conv3_2)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2_relu)
        conv3_3_1x1_reduce_bn = self.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_relu = F.relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3_pad = F.pad(conv3_3_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv3_3_3x3     = self.conv3_3_3x3(conv3_3_3x3_pad)
        conv3_3_3x3_bn  = self.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_relu = F.relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3_relu)
        conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3         = conv3_2_relu + conv3_3_1x1_increase_bn
        conv3_3_relu    = F.relu(conv3_3)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3_relu)
        conv3_4_1x1_reduce_bn = self.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_relu = F.relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3_pad = F.pad(conv3_4_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv3_4_3x3     = self.conv3_4_3x3(conv3_4_3x3_pad)
        conv3_4_3x3_bn  = self.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_relu = F.relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3_relu)
        conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4         = conv3_3_relu + conv3_4_1x1_increase_bn
        conv3_4_relu    = F.relu(conv3_4)
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4_relu)
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4_relu)
        conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1_1x1_reduce_bn = self.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_relu = F.relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3_pad = F.pad(conv4_1_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv4_1_3x3     = self.conv4_1_3x3(conv4_1_3x3_pad)
        conv4_1_3x3_bn  = self.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_relu = F.relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3_relu)
        conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1         = conv4_1_1x1_increase_bn + conv4_1_1x1_proj_bn
        conv4_1_relu    = F.relu(conv4_1)
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1_relu)
        conv4_2_1x1_reduce_bn = self.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_relu = F.relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3_pad = F.pad(conv4_2_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv4_2_3x3     = self.conv4_2_3x3(conv4_2_3x3_pad)
        conv4_2_3x3_bn  = self.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_relu = F.relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3_relu)
        conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2         = conv4_1_relu + conv4_2_1x1_increase_bn
        conv4_2_relu    = F.relu(conv4_2)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2_relu)
        conv4_3_1x1_reduce_bn = self.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_relu = F.relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3_pad = F.pad(conv4_3_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv4_3_3x3     = self.conv4_3_3x3(conv4_3_3x3_pad)
        conv4_3_3x3_bn  = self.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_relu = F.relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3_relu)
        conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3         = conv4_2_relu + conv4_3_1x1_increase_bn
        conv4_3_relu    = F.relu(conv4_3)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3_relu)
        conv4_4_1x1_reduce_bn = self.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_relu = F.relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3_pad = F.pad(conv4_4_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv4_4_3x3     = self.conv4_4_3x3(conv4_4_3x3_pad)
        conv4_4_3x3_bn  = self.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_relu = F.relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3_relu)
        conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4         = conv4_3_relu + conv4_4_1x1_increase_bn
        conv4_4_relu    = F.relu(conv4_4)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4_relu)
        conv4_5_1x1_reduce_bn = self.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_relu = F.relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3_pad = F.pad(conv4_5_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv4_5_3x3     = self.conv4_5_3x3(conv4_5_3x3_pad)
        conv4_5_3x3_bn  = self.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_relu = F.relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3_relu)
        conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5         = conv4_4_relu + conv4_5_1x1_increase_bn
        conv4_5_relu    = F.relu(conv4_5)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5_relu)
        conv4_6_1x1_reduce_bn = self.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_relu = F.relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3_pad = F.pad(conv4_6_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv4_6_3x3     = self.conv4_6_3x3(conv4_6_3x3_pad)
        conv4_6_3x3_bn  = self.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_relu = F.relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3_relu)
        conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6         = conv4_5_relu + conv4_6_1x1_increase_bn
        conv4_6_relu    = F.relu(conv4_6)
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6_relu)
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6_relu)
        conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1_1x1_reduce_bn = self.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_relu = F.relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3_pad = F.pad(conv5_1_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv5_1_3x3     = self.conv5_1_3x3(conv5_1_3x3_pad)
        conv5_1_3x3_bn  = self.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_relu = F.relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3_relu)
        conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1         = conv5_1_1x1_increase_bn + conv5_1_1x1_proj_bn
        conv5_1_relu    = F.relu(conv5_1)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1_relu)
        conv5_2_1x1_reduce_bn = self.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_relu = F.relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3_pad = F.pad(conv5_2_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv5_2_3x3     = self.conv5_2_3x3(conv5_2_3x3_pad)
        conv5_2_3x3_bn  = self.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_relu = F.relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3_relu)
        conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2         = conv5_1_relu + conv5_2_1x1_increase_bn
        conv5_2_relu    = F.relu(conv5_2)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2_relu)
        conv5_3_1x1_reduce_bn = self.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_relu = F.relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3_pad = F.pad(conv5_3_1x1_reduce_relu, (1L, 1L, 1L, 1L))
        conv5_3_3x3     = self.conv5_3_3x3(conv5_3_3x3_pad)
        conv5_3_3x3_bn  = self.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_relu = F.relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3_relu)
        conv5_3_1x1_increase_bn = self.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3         = conv5_2_relu + conv5_3_1x1_increase_bn
        conv5_3_relu    = F.relu(conv5_3)
        pool5_7x7_s1    = F.avg_pool2d(conv5_3_relu, kernel_size=(7L, 7L), stride=(1L, 1L), padding=(0L,), ceil_mode=False)
        feat_extract_0  = pool5_7x7_s1.view(pool5_7x7_s1.size(0), -1)
        feat_extract_1  = self.feat_extract_1(feat_extract_0)
        classifier_8631_f_0 = feat_extract_1.view(feat_extract_1.size(0), -1)
        classifier_8631_f_1 = self.classifier_8631_f_1(classifier_8631_f_0)
        #return classifier_8631_f_1
        return feat_extract_1


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
