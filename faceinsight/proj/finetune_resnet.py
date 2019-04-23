# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function

import os
import numpy as np
import imp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import faceinsight.models.resnet50_128 as resnet_model
from sklearn.metrics import confusion_matrix
from bnuclfdataset import MBTIFaceDataset
#from bnuclfdataset import PF16FaceDataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class FineTuneModel(nn.Module):
    def __init__(self, base_model, class_num):
        super(FineTuneModel, self).__init__()
        self.features = nn.Sequential(*list(base_model.children()))
        base_feat_dim = base_model.feat_extract.out_channels
        self.fc1 = nn.Linear(base_feat_dim, 128)
        self.drop1 = nn.Dropout(p=0.5)
        self.output = nn.Linear(128, class_num)
        
        # freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f= f.view(f.size[0], -1)
        f = relu(self.fc1(f))
        f = self.drop1(f)
        return F.log_softmax(self.output(f), dim=1)

class CNNNet1(nn.Module):
    def __init__(self, class_num):
        super(CNNNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, stride=2)
        self.conv4_bn = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96, 64)
        self.drop1 = nn.Dropout(p=0.5)
        self.output = nn.Linear(64, class_num)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 6, 1)
        x = self.conv4_bn(x)
        x = x.view(-1, 1*1*96)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        return F.log_softmax(self.output(x), dim=1)


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
    #csv_file = os.path.join(data_dir, 'sel_16pf_factors.csv')
    face_dir = os.path.join(data_dir, 'faces')

    # get image stats
    #m, s = get_img_stats(csv_file, face_dir, batch_size=batch_size, 
    #                     num_workers=num_workers, pin_memory=pin_memory)
    
    # define transforms
    #normalize = transforms.Normalize(mean=[0.518, 0.493, 0.506],
    #                                 std=[0.270, 0.254, 0.277])
    normalize = transforms.Normalize(mean=[0.518, 0.493, 0.506],
                                     std=[0.270, 0.254, 0.277])
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(224),
    #transforms.RandomResizedCrop(224, scale=(0.7, 0.9), ratio=(1.0, 1.0)),
    train_transform = transforms.Compose([trasnforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          normalize])
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    # load the dataset
    #train_dataset = MBTIFaceDataset(csv_file, face_dir, 'JP',
    #                                gender_filter=None,
    #                                factor_range=[(0, 11), (18, 23)],
    #                                range2group=True,
    #                                gender2group=False,
    #                                transform=train_transform)
    #test_dataset = MBTIFaceDataset(csv_file, face_dir, 'JP',
    #                               gender_filter=None,
    #                               factor_range=[(0, 11), (18, 23)],
    #                               range2group=True,
    #                               gender2group=False,
    #                               transform=test_transform)
    train_dataset = MBTIFaceDataset(csv_file, face_dir, 'EI', 2500,
                                    class_target=True,
                                    gender_filter=None,
                                    transform=train_transform)
    test_dataset = MBTIFaceDataset(csv_file, face_dir, 'EI', 2500,
                                   class_target=True,
                                   gender_filter=None,
                                   transform=train_transform)
    #train_dataset = PF16FaceDataset(csv_file, face_dir, 'A', 2500,
    #                                class_target=True,
    #                                gender_filter=None,
    #                                transform=train_transform)
    #test_dataset = PF16FaceDataset(csv_file, face_dir, 'A', 2500,
    #                               class_target=True,
    #                               gender_filter=None,
    #                               transform=train_transform)

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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        #output = F.log_softmax(output, dim=1)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.sampler.indices),
                100.*batch_idx*len(data)/len(train_loader.sampler.indices),
                loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_pred = []
    all_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=False)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target).sum().item()
            all_pred.append(pred.cpu().data.numpy())
            all_true.append(target.cpu().data.numpy())

    test_loss /= len(test_loader.sampler.indices)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(test_loader.sampler.indices),
        100.*correct/len(test_loader.sampler.indices)))
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred))
    print(cm*1.0 / cm.sum(axis=1, keepdims=True))
    print('\n')

    return 100.*correct/len(test_loader.sampler.indices)

def run_model(random_seed):
    """Main function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    # load base model
    model_dir = os.path.split(resnet_model.__file__)[0]
    model_def_file = os.path.join(model_dir, 'resnet50_128.py')
    model_weight_file = os.path.join(model_dir, 'resnet50_128.pth')
    MainModel = imp.load_source('MainModel', model_def_file)
    resnet_base = torch.load(model_weight_file)
    
    # model definition
    model = FineTuneModel(resnet_base, 2).to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    test_acc = []
    for epoch in range(1, 31):
        train(net, device, train_loader, optimizer, epoch)
        acc = test(net, device, test_loader)
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

