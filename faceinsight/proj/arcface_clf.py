# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bnudataset import MBTIFaceDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from faceinsight.models.face_activation import Arcface, l2_norm


class CNNNet1(nn.Module):
    def __init__(self):
        super(CNNNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=7, stride=3)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(96, 128, kernel_size=5, stride=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(512, 512, bias=False)
        self.fc1_bn = nn.BatchNorm1d(512)
        #self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3_bn(x)
        x = x.view(-1, 2*2*128)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = l2_norm(x)
        #x = self.drop2(x)
        return x

class CNNNet2(nn.Module):
    def __init__(self):
        super(CNNNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(96, 128, kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(512, 512, bias=False)
        #self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3_bn(x)
        x = x.view(-1, 2*2*128)
        x = self.fc1(x)
        x = l2_norm(x)
        #x = self.drop2(x)
        return x

class CNNNet3(nn.Module):
    def __init__(self):
        super(CNNNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 256, bias=False)
        #self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3_bn(x)
        x = x.view(-1, 2*2*64)
        x = self.fc1(x)
        x = l2_norm(x)
        #x = self.drop2(x)
        return x



def separate_bn_paras(modules):
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules.modules():
        if '__main__' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                for p in layer.parameters():
                    paras_only_bn.append(p)
            else:
                for p in layer.parameters():
                    paras_wo_bn.append(p)
    return paras_only_bn, paras_wo_bn


def get_img_stats(csv_file, face_dir, batch_size, num_workers, pin_memory):
    """Get mean and std. of the images."""
    ds = MBTIFaceDataset(csv_file, face_dir, 'EI', transform=transforms.ToTensor())
    ds_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory,
                                            shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0
    nb_batches = 0
    for data, _ in ds_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        nb_batches += 1
    print('%s batches in total'%(nb_batches))
    print('%s samples in total'%(nb_samples))

    mean /= nb_samples
    std /= nb_samples
    print('mean and std. of the images: %s, %s'%(mean.numpy(), std.numpy()))

    return mean, std

def load_data(data_dir, batch_size, random_seed, test_size=0.1,
              shuffle=True, num_workers=0, pin_memory=False):
    """Utility function for loading and returning train and test
    multi-process iterators over the images.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - test_size: percentage split of the whole dataset used for test set.
      Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/test indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - test_loader: test set iterator.

    """
    error_msg = '[!] test_size should be in the range [0, 1].'
    assert ((test_size>=0) and (test_size<=1)), error_msg

    csv_file = os.path.join(data_dir, 'mbti_factors.csv')
    face_dir = os.path.join(data_dir, 'faces')

    # get image stats
    #m, s = get_img_stats(csv_file, face_dir, batch_size=batch_size, 
    #                     num_workers=num_workers, pin_memory=pin_memory)
    
    # define transforms
    normalize = transforms.Normalize(mean=[0.518, 0.493, 0.506],
                                     std=[0.270, 0.254, 0.277])
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(224),
    #transforms.RandomResizedCrop(224, scale=(0.7, 0.9), ratio=(1.0, 1.0)),
    #train_transform = transforms.Compose([transforms.Resize(112),
    #                                      transforms.ToTensor(),
    #                                      normalize])
    #test_transform = transforms.Compose([transforms.Resize(112),
    #                                     transforms.ToTensor(),
    #                                     normalize])
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          normalize])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         normalize])

    # load the dataset
    train_dataset = MBTIFaceDataset(csv_file, face_dir, 'SN',
                                    gender_filter=None,
                                    factor_range=[(0, 12), (17, 27)],
                                    range2group=True,
                                    transform=train_transform)
    test_dataset = MBTIFaceDataset(csv_file, face_dir, 'SN',
                                   gender_filter=None,
                                   factor_range=[(0, 12), (17, 27)],
                                   range2group=True,
                                   transform=test_transform)

    data_num = len(train_dataset)
    indices = range(data_num)
    split = int(np.floor(data_num * test_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[split:])
    test_sampler = SubsetRandomSampler(indices[:split])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
 
    return (train_loader, test_loader)

def train(model, archead, device, train_loader, optimizer, epoch):
    model.train()
    #archead.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        thetas = archead(embeddings, target)
        loss = F.cross_entropy(thetas, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.sampler.indices),
                100.*batch_idx*len(data)/len(train_loader.sampler.indices),
                loss.item()))

def test(model, archead, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            embeddings = model(data)
            # drop +m part in arcface loss while eval
            kernel_norm = l2_norm(archead.kernel, axis=0)
            thetas = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
            #print(thetas)
            thetas = thetas * 64.
            #thetas = archead(embeddings, target)
            # sum up batch loss
            test_loss += F.cross_entropy(thetas, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = thetas.argmax(dim=1, keepdim=False)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.sampler.indices)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler.indices),
        100.*correct/len(test_loader.sampler.indices)))

    return 100.*correct/len(test_loader.sampler.indices)

def run_model(random_seed):
    """Main function."""
    device = torch.device('cuda:0')

    # load data
    data_dir = '/home/huanglj/proj'

    print('Random seed is %s'%(random_seed))
    train_loader, test_loader = load_data(data_dir,
                                          batch_size=100,
                                          random_seed=random_seed,
                                          test_size=0.1,
                                          shuffle=True,
                                          num_workers=16,
                                          pin_memory=True)

    #model = CNNNet1().to(device)
    model = CNNNet3().to(device)
    archead = Arcface(embedding_size=256, class_num=2, s=64., m=0.5).to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    paras_only_bn, paras_wo_bn = separate_bn_paras(model)
    #print([p.data.shape for p in paras_only_bn])
    #print([p.data.shape for p in paras_wo_bn])
    optimizer = optim.Adam([{'params': paras_wo_bn + [archead.kernel],
                             'weight_decay': 1e-4},
                            {'params': paras_only_bn}],
                           lr=0.001)

    test_acc = []
    for epoch in range(1, 31):
        train(model, archead, device, train_loader, optimizer, epoch)
        acc = test(model, archead, device, test_loader)
        test_acc.append(acc)

    # save test accruacy
    with open('test_acc.csv', 'a+') as f:
        f.write(','.join([str(item) for item in test_acc])+'\n')

def main():
    """Main function."""
    seeds = [10, 25, 69, 30, 22, 91, 65, 83, 11, 8]
    #seeds = [10]
    #random_seed = np.random.randint(100)
    for i in seeds:
        run_model(i)


if __name__=='__main__':
    main()

