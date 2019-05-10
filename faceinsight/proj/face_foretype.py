# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function

import os
import numpy as np
import random
import torch
from bnuclfdataset import PF16FaceDataset
from torchvision import transforms


def run_model(random_seed):
    """Main function."""

    # load data for cross-validation
    data_dir = '/home/huanglj/proj'
    csv_file = os.path.join(data_dir, 'sel_16pf_factors.csv')
    face_dir = os.path.join(data_dir, 'faces')
    sample_size_per_class = 1500
 
    # define transforms
    test_transform = transforms.Compose([transforms.Resize(250),
                                         transforms.CenterCrop(227),
                                         transforms.ToTensor()])

    # load the dataset
    ds = PF16FaceDataset(csv_file, face_dir, 'A',
                         sample_size_per_class,
                         class_target=True,
                         gender_filter=None,
                         transform=test_transform)

    ds_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=100,
                                            num_workers=25,
                                            pin_memory=False,
                                            shuffle=False)

    mean = np.zeros((3, 227, 227))
    nb_samples = 0
    for data, _ in ds_loader:
        nb_samples += data.size(0)
        data = data.sum(0)
        print(data.size)





if __name__=='__main__':
    main()

