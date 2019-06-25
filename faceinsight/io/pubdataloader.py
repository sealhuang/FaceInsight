# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Dataset utils for loading public dataset."""

from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
from PIL import Image

def get_lfw_val_pair(pair_file, img_dir):
    """Get LFW data for validation."""
    pair_info = open(pair_file, 'r').readlines()
    # pop the first line out
    pair_info.pop(0)
    pair_info = [line.strip().split('\t') for line in pair_info]
 
    # data containers
    val_imgs = []
    val_labels = []
    for line in set_info:
        # same pair
        if len(line)==3:
            img1 = os.path.join(img_dir, line[0],
                                '%s_%04d.jpg'%(line[0], line[1]))
            img2 = os.path.join(img_dir, line[0],
                                '%s_%04d.jpg'%(line[0], line[2]))
            if os.path.exists(img1) and os.path.exists(img2):
                val_imgs.append(img1)
                val_imgs.append(img2)
                val_labels.append(1)
        # different pair
        elif len(line)==4:
            img1 = os.path.join(img_dir, line[0],
                                '%s_%04d.jpg'%(line[0], line[1]))
            img2 = os.path.join(img_dir, line[2],
                                '%s_%04d.jpg'%(line[2], line[3]))
            if os.path.exists(img1) and os.path.exists(img2):
                val_imgs.append(img1)
                val_imgs.append(img2)
                val_labels.append(0)
        assert len(val_imgs)==len(val_labels)*2, 'Unmatch data pair'
        print('%s pairs collected'%(len(val_labels)))

    return val_imgs, np.array(val_labels)

