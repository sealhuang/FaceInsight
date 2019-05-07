# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from bnuclfdataset import PF16MultiLabelDataset
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

class CNNNet1(nn.Module):
    def __init__(self, target_info):
        super(CNNNet1, self).__init__()
        self.target_info = target_info
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, bias=True)
        self.conv2 = nn.Conv2d(32, 96, kernel_size=5, stride=2, bias=True)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=3, bias=True)
        self.conv4_bn = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96, 96, bias=True)
        self.drop1 = nn.Dropout(p=0.5)
        for c in target_info:
            setattr(self, 'output_%s'%(c), nn.Linear(96, target_info[c],
                                                     bias=True))

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4_bn(x)
        x = x.view(-1, 1*1*96)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        outs = list()
        dir(self)
        for c in self.target_info:
            fun = eval('self.output_%s'%(c))
            out = F.log_softmax(fun(x), dim=1)
            outs.append(out)
        return outs

class CNNNet4(nn.Module):
    def __init__(self, class_num):
        super(CNNNet4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, bias=True)
        self.conv2 = nn.Conv2d(32, 96, kernel_size=5, stride=2, bias=True)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=3, bias=True)
        self.conv4_bn = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96, 96, bias=True)
        self.drop1 = nn.Dropout(p=0.5)
        self.output = nn.Linear(96, class_num, bias=True)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
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

def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(len(output))
        print(output)
        # calculate multilabel loss
        loss = torch.FloatTensor(1).zero_().to(device)
        loss_list = list()
        for i in range(len(output)):
            sub_loss = F.nll_loss(output[i], target[:, i])
            loss_list.append(sub_loss.data[0])
            loss += sub_loss
        loss.backward()
        optimizer.step()
        
        # visualize training loss for each factor
        for i in range(len(output)):
            writer.add_scalar('data/training-loss-%s'%(i+1),
                              loss_list[i],
                (epoch-1)*int(len(train_loader.sampler.indices)/64)+batch_idx+1)
        if batch_idx % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.sampler.indices),
                100.*batch_idx*len(data)/len(train_loader.sampler.indices),
                loss.item()))

def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    all_pred = []
    all_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=False)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target).sum().item()
            all_pred.append(pred.cpu().data.numpy())
            all_true.append(target.cpu().data.numpy())
    
    # plot model parameter hist
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    params = model.state_dict()
    #print(params)
    x = vutils.make_grid(params['conv1.weight'].clone().cpu().data,
                         normalize=True, scale_each=True)
    writer.add_image('Image', x, epoch)

    test_loss /= len(test_loader.sampler.indices)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(test_loader.sampler.indices),
        100.*correct/len(test_loader.sampler.indices)))
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred))
    print(cm*1.0 / cm.sum(axis=1, keepdims=True))
    print('\n')
    writer.add_scalar('data/test-accuracy', 100.*correct/len(test_loader.sampler.indices), epoch)

    return 100.*correct/len(test_loader.sampler.indices)

def run_model(random_seed):
    """Main function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data config
    data_dir = '/home/huanglj/proj'
    csv_file = os.path.join(data_dir, 'sel_16pf_factors.csv')
    face_dir = os.path.join(data_dir, 'faces')

    # define transforms
    normalize = transforms.Normalize(mean=[0.518, 0.493, 0.506],
                                     std=[0.270, 0.254, 0.277])
    train_transform = transforms.Compose([
                        transforms.Resize(250),
                        transforms.RandomCrop(227),
                        transforms.ColorJitter(brightness=.05, saturation=.05),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize])
    test_transform = transforms.Compose([
                        transforms.Resize(250),
                        transforms.CenterCrop(227),
                        transforms.ToTensor(),
                        normalize])

    # load dataset
    train_dataset = PF16MultilabelDataset(csv_file, face_dir,
                                          ['A', 'E', 'F', 'H', 'L', 'N'],
                                          gender_filter=None,
                                          transform=train_transform)
    test_dataset = PF16MultilabelDataset(csv_file, face_dir,
                                         ['A', 'E', 'F', 'H', 'L', 'N'],
                                         gender_filter=None,
                                         transform=test_transform)

    # split dataset
    test_ratio = 0.2
    sample_idx = range(len(train_dataset))
    split_idx = int(np.floor(len(train_dataset) * test_ratio))
    # get training- and testing-samples
    print('Random seed is %s'%(random_seed))
    np.random.seed(random_seed)
    np.random.shuffle(sample_idx)
    # CV
    #for fold in range(int(1/test_ratio)):
    for fold in range(1):
        print('Fold %s/%s'%(fold+1, int(1/test_ratio)))
        train_sampler = SubsetRandomSampler(sample_idx[split_idx:])
        test_sampler = SubsetRandomSampler(sample_idx[:split_idx])
        sample_idx = sample_idx[split_idx:] + sample_idx[:split_idx]
        # get dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   sampler=train_sampler,
                                                   num_workers=25,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  sampler=test_sampler,
                                                  num_workers=25,
                                                  pin_memory=True)
 
        # model config
        model = CNNNet1(train_dataset.target_info).to(device)
        # summary writer config
        writer = SummaryWriter()
        #writer.add_graph(CNNNet3(2))

        #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        test_acc = []
        for epoch in range(1, 51):
            train(model, device, train_loader, optimizer, epoch, writer)
            acc = test(model, device, test_loader, epoch, writer)
            test_acc.append(acc)

        # save test accruacy
        with open('test_acc.csv', 'a+') as f:
            f.write(','.join([str(item) for item in test_acc])+'\n')
    
        #writer.export_scalars_to_json('./all_scalars_%s.json'%(random_seed))
        writer.close()


def main():
    """Main function."""
    #seeds = [10, 25, 69, 30, 22, 91, 65, 83, 11, 8]
    seeds = [10]
    for i in seeds:
        run_model(i)
    
if __name__=='__main__':
    main()

