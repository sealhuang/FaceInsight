# vi: set ft=python sts=4 ts=4 sw=4 et:

import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils

from faceinsight.models.shufflenet_v2 import ShuffleNetV2

from config import configurations


if __name__ == '__main__':

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]
 
    BACKBONE_NAME = cfg['BACKBONE_NAME'] 
    INPUT_SIZE = cfg['INPUT_SIZE']
    # for normalize inputs
    RGB_MEAN = cfg['RGB_MEAN']
    RGB_STD = cfg['RGB_STD']
    # feature dimension
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']

    DEVICE = 'cpu'

    test_transform = transforms.Compose([
                        transforms.Resize(250),
                        transforms.CenterCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
                        ])
    test_data_dir = os.path.join('/Users/sealhuang/Downloads/test_imgs')
    dataset_test = datasets.ImageFolder(test_data_dir, test_transform)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=False,
                                              num_workers=1,
                                              drop_last=False)

    NUM_CLASS = len(test_loader.dataset.classes)
    print("Number of Testing Classes: {}".format(NUM_CLASS))

    # ======= model ======= #
    model_file = 'Backbone_shufflenet_v2_x1_0_Epoch_16_Batch_15280_Time_2019-07-25-18-51_checkpoint.pth'
    if BACKBONE_NAME=='shufflenet_v2_x0_5':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=INPUT_SIZE[0],
                                width_mult=0.5)
    elif BACKBONE_NAME=='shufflenet_v2_x1_0':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=INPUT_SIZE[0],
                                width_mult=1.0)
        BACKBONE.load_state_dict(torch.load(model_file,
                                    map_location=lambda storage, loc: storage))
        if DEVICE=='gpu':
            BACKBONE = BACKBONE.cuda()
        BACKBONE.eval()
    elif BACKBONE_NAME=='shufflenet_v2_x1_5':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=INPUT_SIZE[0],
                                width_mult=1.5)
    elif BACKBONE_NAME=='shufflenet_v2_x2_0':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=INPUT_SIZE[0],
                                width_mult=2.0)
    elif BACKBONE_NAME=='mobilefacenet':
        BACKBONE = MobileFaceNet(EMBEDDING_SIZE)
    else:
        pass

    print('=' * 60)
    print(BACKBONE)
    print('{} Backbone Generated'.format(BACKBONE_NAME))
    print('=' * 60)

    test_vtrs = []

    for inputs, labels in tqdm(iter(test_loader)):
        # compute output
        inputs = inputs.to(DEVICE)
        features = BACKBONE(inputs)
        #print(features.cpu().data.numpy().shape)
        test_vtrs.append(features.cpu().data.numpy())

    test_vtrs = np.concatenate(tuple(test_vtrs))
    print(test_vtrs.shape)
    np.save('eval_feats.npy', test_vtrs)

