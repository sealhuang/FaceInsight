# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Utils for reading image data via MXNet."""

import os
import pickle

import numpy as np
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import bcolz
from tqdm import tqdm
import mxnet as mx


def load_bin(bin_file, root_dir, image_size=[112, 112]):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, mode=0o755)
    bins, issame_list = pickle.load(open(bin_file, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]],
                      dtype=np.float32,
                      rootdir=root_dir,
                      mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(root_dir) + '_list', np.array(issame_list))
    return data, issame_list

def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'imgs')
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o755)
    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'),
                                           os.path.join(rec_path, 'train.rec'),
                                           'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)
        label_path = os.path.join(save_path, str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path, mode=0o755)
        img.save(os.path.join(label_path, '{}.jpg'.format(idx)), quality=95)

