# -*- coding: utf8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
from PIL import Image
from torch.utils.data import Dataset


def check_factor_range_validity(factor_range):
    if isinstance(factor_range, list):
        for t in factor_range:
            if isinstance(t, tuple):
                if len(t)==2 and t[1]>t[0]:
                    pass
                else:
                    raise(RuntimeError('Invalid factor range: %s'%(t)))
            else:
                raise(RuntimeError('Tuple in factor_range is required.'))
    else:
        raise(RuntimeError('List of factor range is required.'))

def make_dataset(csv_info, img_dir, target_idx,
                 factor_range=None, range2group=False):
    samples = []

    # target preprocessing
    if factor_range:
        # check the validity of factor range
        check_factor_range_validity(factor_range)
        # determine the target type
        if isinstance(range2group, bool) and range2group:
            if len(factor_range)>1:
                label_dict = dict(zip([str(t) for t in factor_range],
                                      range(len(factor_range))))
                print(label_dict)
            else:
                raise(RuntimeError('Only one factor range in the list.'))
        else:
            print('Non-categorical target output.')

    # sample selection
    for line in csv_info:
        img = os.path.join(img_dir, line[-1])
        if not os.path.exists(img):
            print('Non-exist image: %s'%(img))
            continue
        v = float(line[target_idx])
        if factor_range:
            for t in factor_range:
                if v>=t[0] and v<t[1]:
                    if range2group:
                        v = label_dict[str(t)]
                    samples.append((img, v))
                    break
        else:
            samples.append((img, v))

    return samples

class MBTIFaceDataset(Dataset):
    """MBTI data loader.
    
    Args:
        csv_file (string): MBTI data file.
        img_dir (string): Directory path of face images.
        factor_name (string): target factor name.
        gender_filter (string): 'male', 'female', or None for default.
        factor_range (list of tuples): e.g. [(0, 8), (15, 24)]
        range2group (boolean): False for default.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g. ``transforms.RandomCrop`` for images.
        target_transform (calllable, optional): A function/transform that takes
            in the target and transforms it.

    """
    def __init__(self, csv_file, img_dir, factor_name, gender_filter=None,
                 factor_range=None, range2group=False,
                 transform=None, target_transform=None):
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
                               factor_range=factor_range,
                               range2group=range2group)
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

