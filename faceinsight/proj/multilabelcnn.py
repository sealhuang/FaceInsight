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

class CNNNet2(nn.Module):
    def __init__(self, target_info):
        super(CNNNet2, self).__init__()
        self.target_info = target_info
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, bias=True)
        self.conv2 = nn.Conv2d(32, 96, kernel_size=5, stride=1,
                               padding=2, bias=True)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=5, stride=2, bias=True)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, stride=3, bias=True)
        self.conv4_bn = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96, 96, bias=True)
        self.drop1 = nn.Dropout(p=0.5)
        #self.fc2 = nn.Linear(96, 96, bias=True)
        for c in target_info:
            setattr(self, 'output_%s'%(c), nn.Linear(96, target_info[c],
                                                     bias=True))

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4_bn(x)
        x = x.view(-1, 1*1*96)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        #x = F.relu(self.fc2(x))
        outs = list()
        dir(self)
        for c in self.target_info:
            fun = eval('self.output_%s'%(c))
            out = F.log_softmax(fun(x), dim=1)
            outs.append(out)
        return outs

class CNNNet3(nn.Module):
    def __init__(self, target_info):
        super(CNNNet3, self).__init__()
        self.target_info = target_info
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, bias=True)
        self.conv2 = nn.Conv2d(32, 96, kernel_size=5, stride=1,
                               padding=2, bias=True)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=5, stride=2, bias=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=3, bias=True)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 128, bias=True)
        self.drop1 = nn.Dropout(p=0.3)
        for c in target_info:
            setattr(self, 'output_%s'%(c), nn.Linear(128, target_info[c],
                                                     bias=True))

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4_bn(x)
        x = x.view(-1, 1*1*128)
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
    def __init__(self, target_info):
        super(CNNNet4, self).__init__()
        self.target_info = target_info
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, bias=True)
        self.conv2 = nn.Conv2d(32, 96, kernel_size=5, stride=1, padding=2,
                               bias=True)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, stride=3, bias=True)
        self.conv4_bn = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(384, 128, bias=True)
        self.drop1 = nn.Dropout(p=0.5)
        for c in target_info:
            setattr(self, 'output_%s'%(c), nn.Linear(128, target_info[c],
                                                     bias=True))

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv4(x))
        x = self.conv4_bn(x)
        x = x.view(-1, 2*2*96)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        outs = list()
        dir(self)
        for c in self.target_info:
            fun = eval('self.output_%s'%(c))
            out = F.log_softmax(fun(x), dim=1)
            outs.append(out)
        return outs


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
        # calculate multilabel loss
        loss = torch.FloatTensor(1).zero_().to(device)
        loss_list = list()
        for i in range(len(output)):
            sub_loss = F.nll_loss(output[i], target[:, i])
            loss_list.append(sub_loss.item())
            loss += sub_loss
        loss.backward()
        optimizer.step()
        
        # visualize training loss for each factor
        for i in range(len(output)):
            writer.add_scalar('data/training-loss-%s'%(i+1),
                              loss_list[i],
                (epoch-1)*int(len(train_loader.sampler.indices)/64)+batch_idx+1)
        if batch_idx % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.sampler.indices),
                100.*batch_idx*len(data)/len(train_loader.sampler.indices),
                loss.item()))

def test(model, device, test_loader, epoch, writer):
    model.eval()
    loss = 0
    # loss for each factor
    loss_list = []
    for i in range(len(model.target_info)):
        loss_list.append(0)
    # correct prediction
    correct_list = []
    for i in range(len(model.target_info)):
        correct_list.append(0)
    # pred- and true-item for confusion matrix
    all_pred_list = []
    for i in range(len(model.target_info)):
        all_pred_list.append([])
    all_true_list = []
    for i in range(len(model.target_info)):
        all_true_list.append([])

    # model eval
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # get test metrics
            # sum up batch loss
            for i in range(len(output)):
                sub_loss = F.nll_loss(output[i], target[:, i],
                                      reduction='sum').item()
                loss_list[i] += sub_loss
                loss += sub_loss
                # get the index of the max-log-probability
                pred = output[i].argmax(dim=1, keepdim=False)
                correct_list[i] += pred.eq(target[:, i]).sum().item()
                all_pred_list[i].append(pred.cpu().data.numpy())
                all_true_list[i].append(target[:, i].cpu().data.numpy())
    
    # plot model parameter hist
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    params = model.state_dict()
    x = vutils.make_grid(params['conv1.weight'].clone().cpu().data,
                         normalize=True, scale_each=True)
    writer.add_image('Image', x, epoch)

    loss /= len(test_loader.sampler.indices)
    print('Test set: Average loss: {:.4f}'.format(loss))
    
    # print confusion matrix
    for i in range(len(model.target_info)):
        cm = confusion_matrix(np.concatenate(all_true_list[i]),
                              np.concatenate(all_pred_list[i]))
        print(cm*1.0 / cm.sum(axis=1, keepdims=True))
        print('------------------------------------------\n')
        # add scalar plot
        writer.add_scalar('data/test-accuracy-%s'%(i+1),
                          100.*correct_list[i]/len(test_loader.sampler.indices),
                          epoch)

    return [100.*correct_list[i]/len(test_loader.sampler.indices)
                for i in range(len(correct_list))]

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
    train_dataset = PF16MultiLabelDataset(csv_file, face_dir,
                                          ['A', 'E', 'H', 'L', 'N'],
                                          gender_filter=None,
                                          transform=train_transform)
    test_dataset = PF16MultiLabelDataset(csv_file, face_dir,
                                         ['A', 'E', 'H', 'L', 'N'],
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
    print('\nStart Cross-Validation ........')
    for fold in range(int(1/test_ratio)):
    #for fold in range(1):
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
        #model = CNNNet1(train_dataset.target_info).to(device)
        #model = CNNNet2(train_dataset.target_info).to(device)
        model = CNNNet4(train_dataset.target_info).to(device)
        # summary writer config
        writer = SummaryWriter()
        #writer.add_graph(CNNNet3(2))

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=140,
                                              gamma=0.5)
        #optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)

        # test accuracy init
        test_acc = []
        for i in range(len(train_dataset.target_info)):
            test_acc.append([])
        for epoch in range(1, 501):
            scheduler.step(epoch)
            train(model, device, train_loader, optimizer, epoch, writer)
            sub_acc = test(model, device, test_loader, epoch, writer)
            for i in range(len(sub_acc)):
                test_acc[i].append(sub_acc[i])

        # save test accruacy
        for i in range(len(test_acc)):
            with open('test_acc_multilabel_%s.csv'%(i+1), 'a+') as f:
                f.write(','.join([str(item) for item in test_acc[i]])+'\n')
    
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

