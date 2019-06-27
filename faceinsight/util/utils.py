# vi: set ft=python sts=4 sw=4 ts=4 et:

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import io
from datetime import datetime

import bcolz
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F

from .verification import evaluate


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')

def make_weights_for_balanced_classes(images, nclasses):
    """Make a vector of weights for each image in the dataset, based
    on class frequency. The returned vector of weights can be used
    to create a WeightedRandomSampler for a DataLoader to have
    class balancing when sampling for a training batch.
        images - torchvisionDataset.imgs
        nclasses - len(torchvisionDataset.classes)
    ref: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    """
    count = [0] * nclasses
    # item is (img-data, label-id)
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    # total number of images
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))
    return carray, issame

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up
    # print(optimizer)

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.
    print(optimizer)

def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

hflip = transforms.Compose([de_preprocess,
                            transforms.ToPILImage(),
                            transforms.functional.hflip,
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5])
                        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs

ccrop = transforms.Compose([de_preprocess,
                            transforms.ToPILImage(),
                            transforms.Resize([256, 256]),
                            transforms.CenterCrop([224, 224]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5])
                        ])

def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, carray,
                issame, nrof_folds=10, tta=True):
    if multi_gpu:
        # unpackage model from DataParallel
        backbone = backbone.module
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    # switch to evaluation mode
    backbone.eval()

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx+batch_size][:, [2, 1, 0], :, :])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + \
                            backbone(fliped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                emb_batch = backbone(ccropped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + \
                            backbone(fliped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                emb_batch = backbone(ccropped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings,issame,nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold,epoch)
    writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)

def read_imgs(files):
    imgs = []
    for item in files:
        with open(item, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img = to_tensor(img)
            img = (img - 0.5) / 0.5
            imgs.append(img)

    return torch.stack(imgs, dim=0)

def perform_lfw_val(multi_gpu, device, embedding_size, batch_size, backbone,
                    imgs, issame, nrof_folds=10, tta=True):
    if multi_gpu:
        # unpackage model from DataParallel
        backbone = backbone.module
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    # switch to evaluation mode
    backbone.eval()

    idx = 0
    embeddings = np.zeros([len(imgs), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(imgs):
            batch_files = imgs[idx:idx+batch_size]
            batch = read_imgs(batch_files)
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + \
                            backbone(fliped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                emb_batch = backbone(ccropped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            idx += batch_size
        if idx < len(imgs):
            batch_files = imgs[idx:]
            batch = read_imgs(batch_files)
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + \
                            backbone(fliped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                emb_batch = backbone(ccropped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings,issame,nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

