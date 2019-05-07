# -*- coding: utf8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(csv_info, img_dir, target_idx, sample_num_per_end,
                 class_target=True):
    samples = []
    # sort factor value
    imgs = []
    factors = []
    for line in csv_info:
        img = os.path.join(img_dir, line[-1])
        if not os.path.exists(img):
            print('Non-exist image: %s'%(img))
            continue
        factors.append(float(line[target_idx]))
        imgs.append(img)
    sorted_idx = np.argsort(factors)
    sorted_factors = []
    sorted_imgs = []
    for i in range(len(sorted_idx)):
        sorted_imgs.append(imgs[sorted_idx[i]])
        sorted_factors.append(factors[sorted_idx[i]])

    # select samples
    lower_part_thresh = sorted_factors[sample_num_per_end]
    upper_part_thresh = sorted_factors[-sample_num_per_end]
    print('Select lower end %s samples, threshold %s'%
                (sample_num_per_end, lower_part_thresh))
    print('Select upper end %s samples, threshold %s'%
                (sample_num_per_end, upper_part_thresh))

    # target info
    if class_target:
        label_dict = {'lower': 0, 'upper': 1}
        print(label_dict)

    # sample selection
    for i in range(sample_num_per_end):
        if class_target:
            samples.append((sorted_imgs[i], 0))
        else:
            samples.append((sorted_imgs[i], sorted_factors[i]))
    for i in range(sample_num_per_end):
        if class_target:
            samples.append((sorted_imgs[-i], 1))
        else:
            samples.append((sorted_imgs[-i], sorted_factors[-i]))

    return samples

class MBTIFaceDataset(Dataset):
    """MBTI data loader.
    
    Args:
        csv_file (string): MBTI data file.
        img_dir (string): Directory path of face images.
        factor_name (string): target factor name.
        sample_num_per_end (int): select N samples from each end of the `factor`
            value distribution.
        class_target (boolean): True for default.
        gender_filter (string): 'male', 'female', or None for default.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g. ``transforms.RandomCrop`` for images.
        target_transform (calllable, optional): A function/transform that takes
            in the target and transforms it.

    """
    def __init__(self, csv_file, img_dir,
                 factor_name, sample_num_per_end, class_target=True,
                 gender_filter=None, transform=None, target_transform=None):
        # read csv info and get dataset
        csv_info = open(csv_file).readlines()
        csv_info = [line.strip().split(',') for line in csv_info]
        header = csv_info.pop(0)
        target_idx = header.index(factor_name)
        # gender filter
        if gender_filter:
            if gender_filter not in ['male', 'female']:
                raise(RuntimeError('Invalid gender_filter argument.'))
            csv_info = [line for line in csv_info if line[1]==gender_filter]
        samples = make_dataset(csv_info, img_dir, target_idx,
                               sample_num_per_end=sample_num_per_end,
                               class_target=class_target)
        if len(samples)==0:
            raise(RuntimeError('Found 0 files in face folder.'))

        self.face_dir = os.path.expanduser(img_dir)
        self.target_factor = factor_name
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is factor value of MBTI factor.
        """
        img_path, target = self.samples[idx]
        # load face image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Face dir location : {}\n'.format(self.face_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class PF16FaceDataset(Dataset):
    """16PF data loader.
    
    Args:
        csv_file (string): 16PF data file.
        img_dir (string): Directory path of face images.
        factor_name (string): target factor name.
        sample_num_per_end (int): select N samples from each end of the `factor`
            value distribution.
        class_target (boolean): True for default.
        gender_filter (string): 'male', 'female', or None for default.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g. ``transforms.RandomCrop`` for images.
        target_transform (calllable, optional): A function/transform that takes
            in the target and transforms it.

    """
    def __init__(self, csv_file, img_dir,
                 factor_name, sample_num_per_end, class_target=True,
                 gender_filter=None, transform=None, target_transform=None):
        # read csv info and get dataset
        csv_info = open(csv_file).readlines()
        csv_info = [line.strip().split(',') for line in csv_info]
        header = csv_info.pop(0)
        target_idx = header.index(factor_name)
        # gender filter
        if gender_filter:
            if gender_filter not in ['male', 'female']:
                raise(RuntimeError('Invalid gender_filter argument.'))
            csv_info = [line for line in csv_info if line[1]==gender_filter]
        samples = make_dataset(csv_info, img_dir, target_idx,
                               sample_num_per_end=sample_num_per_end,
                               class_target=class_target)
        if len(samples)==0:
            raise(RuntimeError('Found 0 files in face folder.'))

        self.face_dir = os.path.expanduser(img_dir)
        self.target_factor = factor_name
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is factor value of MBTI factor.
        """
        img_path, target = self.samples[idx]
        # load face image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Face dir location : {}\n'.format(self.face_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def make_multilabel_dataset(csv_info, img_dir, target_idxs):
    imgs = []
    targets = []

    # get image list
    for line in csv_info:
        img = os.path.join(img_dir, line[-1])
        if not os.path.exists(img):
            print('Non-exist image: %s'%(img))
            imgs.append('None')
        else:
            imgs.append(img)

    # get factor targets
    for i in target_idxs:
        print('%s-th factor'%(i))
        factor_vals = [float(line[i]) for line in csv_info]
        # sort factor value
        sorted_idx = np.argsort(factor_vals)
        # convert sorted index to target class
        class_vals = [0 if idx in sorted_idx[:2000]
                      else (1 if idx in sorted_idx[-2000:] else 2)
                      for idx in range(len(factor_vals))]
        print('s% samples in Class 0'%(np.sum(np.array(class_vals)==0)))
        print('s% samples in Class 1'%(np.sum(np.array(class_vals)==1)))
        print('s% samples in Class 2'%(np.sum(np.array(class_vals)==2)))
        targets.append(class_vals)

    # convert targets from rows to columns
    targets = np.vstack(targets)
    targets = targets.T.tolist()
    #print('%s images and %s targets'%(len(imgs), len(targets)))
    assert len(imgs)==len(targets)
 
    return imgs, targets

class PF16MultiLabelDataset(Dataset):
    """16PF data loader.
    
    Args:
        csv_file (string): 16PF data file.
        img_dir (string): Directory path of face images.
        factor_names (list of string): target factor names.
        gender_filter (string): 'male', 'female', or None for default.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g. ``transforms.RandomCrop`` for images.
        target_transform (calllable, optional): A function/transform that takes
            in the target and transforms it.

    """
    def __init__(self, csv_file, img_dir, factor_names,
                 gender_filter=None, transform=None, target_transform=None):
        # read csv info and get dataset
        csv_info = open(csv_file).readlines()
        csv_info = [line.strip().split(',') for line in csv_info]

        # check the factor availablity
        header = csv_info.pop(0)
        target_idxs = []
        target_info = {}
        for n in factor_names:
            try:
                target_idxs.append(header.index(n))
                target_info[n] = 3
            except:
                print('Invalid factor name.')
                raise

        # gender filter
        if gender_filter:
            if gender_filter not in ['male', 'female']:
                raise(RuntimeError('Invalid gender_filter argument.'))
            csv_info = [line for line in csv_info if line[1]==gender_filter]
        imgs, targets = make_multilabel_dataset(csv_info, img_dir, target_idxs)
        if len(imgs)==0:
            raise(RuntimeError('Found 0 files in face folder.'))

        self.face_dir = os.path.expanduser(img_dir)
        self.target_factors = factor_names
        self.imgs = imgs
        self.targets = targets
        self.target_info = target_info

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is factor value of MBTI factor.
        """
        img_path, target = self.imgs[idx], self.targets[idx]
        # load face image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Face dir location : {}\n'.format(self.face_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

