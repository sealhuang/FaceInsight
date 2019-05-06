# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Script to train an AlexNet-like model for facial expression recognition."""

from __future__ import print_function

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter


class AffectNet(nn.Module):
    def __init__(self, class_num):
        super(AffectNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar('data/training-loss', loss,
                          (epoch-1)*int(len(train_loader.sampler.indices)/64)+batch_idx+1)
        if batch_idx % 15 == 0:
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

def run_model():
    """Main function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    data_dir = '/home/huanglj/proj/affectnet'
    # define transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(250),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.05, saturation=.05),
            transforms.ToTensor(),
            normalize]),
        'val': transforms.Compose([
            transforms.Resize(250),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])}
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, '%s_class'%(x)),
                                     data_transforms[x])
             for x in ['train', 'val']}
    print(dsets['train'].classes)
    print(dsets['train'].classes)
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x],
                                                   batch_size=100,
                                                   num_workers=25,
                                                   shuffle=True,
                                                   pin_memory=True)
                    for x in ['train', 'val']}

    # model training and eval
    model = AffectNet().to(device)
    # summary writer config
    writer = SummaryWriter()
    #writer.add_graph(CNNNet3(2))

    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    test_acc = []
    for epoch in range(1, 231):
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
    run_model()


if __name__=='__main__':
    main()

