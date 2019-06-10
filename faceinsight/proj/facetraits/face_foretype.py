# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function

import os
import numpy as np
import random
import torch
from bnuclfdataset import PF16FaceDataset
from torchvision import transforms
import matplotlib.pyplot as plt

def run_model(factor_name):
    """Main function."""

    # load data for cross-validation
    data_dir = '/Users/sealhuang/project/faceTraits/bnuData'
    #data_dir = '/home/huanglj/proj'
    csv_file = os.path.join(data_dir, '16pf_workbench', 'sel_16pf_factors.csv')
    face_dir = os.path.join(data_dir, 'aligned_faces')
    sample_size_per_class = 50
    #factor_name = 'C'
 
    # define transforms
    test_transform = transforms.Compose([transforms.Resize(250),
                                         transforms.CenterCrop(227),
                                         transforms.ToTensor()])

    # load the dataset
    ds = PF16FaceDataset(csv_file, face_dir, factor_name,
                         sample_size_per_class,
                         class_target=True,
                         gender_filter='male',
                         transform=test_transform)

    ds_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=1,
                                            num_workers=25,
                                            pin_memory=False,
                                            shuffle=False)

    c1mean = np.zeros((3, 227, 227))
    c2mean = np.zeros((3, 227, 227))
    c1nb = 0
    c2nb = 0
    for data, target in ds_loader:
        batch_size = data.size(0)
        if target[0]:
            c1mean += data.sum(0).numpy()
            c1nb += batch_size
        else:
            c2mean += data.sum(0).numpy()
            c2nb += batch_size
    c1mean /= c1nb
    c2mean /= c2nb

    # plot image
    f, axarr = plt.subplots(1, 3, sharey=True)
    axarr[0].imshow(np.moveaxis(c1mean, 0, -1))
    axarr[1].imshow(np.moveaxis(c2mean, 0, -1))
    con = np.moveaxis(c1mean, 0, -1) - np.moveaxis(c2mean, 0, -1)
    con = (con - con.min()) / (con.max() - con.min())
    axarr[2].imshow(con)

    f.savefig('male_%s_foretype.png'%(factor_name))

def main():
    factors = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N',
               'O', 'Q1', 'Q2', 'Q3', 'Q4', 'X1', 'X2', 'X3', 'X4',
               'Y1', 'Y2', 'Y3', 'Y4']
    for f in factors:
        run_model(f)


if __name__=='__main__':
    main()

