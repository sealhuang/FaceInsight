# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Dataset utils for building the face recognition network.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
import math


class ImageClass():
    """Stores the paths to images for a given class"""
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images
                        if not img.startswith('.')]
    return image_paths
  
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    if has_class_directories:
        classes = [path for path in os.listdir(path_exp)
                    if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
    else:
        path_name = os.path.basename(path_exp)
        image_paths = get_image_paths(path_exp)
        dataset.append(ImageClass(path_name, image_paths))

    return dataset

#def get_image_paths_and_labels(dataset):
#    image_paths_flat = []
#    labels_flat = []
#    for i in range(len(dataset)):
#        image_paths_flat += dataset[i].image_paths
#        labels_flat += [i] * len(dataset[i].image_paths)
#    return image_paths_flat, labels_flat
#
#def shuffle_examples(image_paths, labels):
#    shuffle_list = list(zip(image_paths, labels))
#    random.shuffle(shuffle_list)
#    image_paths_shuff, labels_shuff = zip(*shuffle_list)
#    return image_paths_shuff, labels_shuff
#
#def get_batch(image_data, batch_size, batch_index):
#    nrof_examples = np.size(image_data, 0)
#    j = batch_index*batch_size % nrof_examples
#    if j+batch_size<=nrof_examples:
#        batch = image_data[j:j+batch_size,:,:,:]
#    else:
#        x1 = image_data[j:nrof_examples,:,:,:]
#        x2 = image_data[0:nrof_examples-j,:,:,:]
#        batch = np.vstack([x1,x2])
#    batch_float = batch.astype(np.float32)
#    return batch_float
#
#def get_label_batch(label_data, batch_size, batch_index):
#    nrof_examples = np.size(label_data, 0)
#    j = batch_index*batch_size % nrof_examples
#    if j+batch_size<=nrof_examples:
#        batch = label_data[j:j+batch_size]
#    else:
#        x1 = label_data[j:nrof_examples]
#        x2 = label_data[0:nrof_examples-j]
#        batch = np.vstack([x1,x2])
#    batch_int = batch.astype(np.int64)
#    return batch_int
#
#def get_triplet_batch(triplets, batch_index, batch_size):
#    ax, px, nx = triplets
#    a = get_batch(ax, int(batch_size/3), batch_index)
#    p = get_batch(px, int(batch_size/3), batch_index)
#    n = get_batch(nx, int(batch_size/3), batch_index)
#    batch = np.vstack([a, p, n])
#    return batch
#
#def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
#    if mode=='SPLIT_CLASSES':
#        nrof_classes = len(dataset)
#        class_indices = np.arange(nrof_classes)
#        np.random.shuffle(class_indices)
#        split = int(round(nrof_classes*(1-split_ratio)))
#        train_set = [dataset[i] for i in class_indices[0:split]]
#        test_set = [dataset[i] for i in class_indices[split:-1]]
#    elif mode=='SPLIT_IMAGES':
#        train_set = []
#        test_set = []
#        for cls in dataset:
#            paths = cls.image_paths
#            np.random.shuffle(paths)
#            nrof_images_in_class = len(paths)
#            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
#            if split==nrof_images_in_class:
#                split = nrof_images_in_class-1
#            if split>=min_nrof_images_per_class and nrof_images_in_class-split>=1:
#                train_set.append(ImageClass(cls.name, paths[:split]))
#                test_set.append(ImageClass(cls.name, paths[split:]))
#    else:
#        raise ValueError('Invalid train/test split mode "%s"' % mode)
#    return train_set, test_set

