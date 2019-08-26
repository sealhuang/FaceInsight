# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from faceinsight.models.shufflefacenet import ShuffleNetV2
from bnuclfdataset import PF16FaceDataset


class clsNet1(nn.Module):
    def __init__(self, class_num):
        super(clsNet1, self).__init__()
        self.fc1 = nn.Linear(512, class_num, bias=False)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = self.fc1(x)
        return x

class clsNet2(nn.Module):
    def __init__(self, class_num):
        super(clsNet2, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(class_num, 512))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine


def load_data(factor, data_dir, sample_size_per_class,
              train_sampler, test_sampler, batch_size, shuffle=True,
              num_workers=0, pin_memory=False):
    """Utility function for loading and returning train and test
    multi-process iterators over the images.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - test_loader: test set iterator.

    """
    csv_file = os.path.join(data_dir, 'sel_16pf_factors.csv')
    face_dir = os.path.join(data_dir, 'bnu_aligned_faces')
 
    # define transforms
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([transforms.Resize(250),
                                          transforms.RandomCrop(224),
                                          transforms.ColorJitter(brightness=.05,
                                                                 saturation=.05),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    test_transform = transforms.Compose([transforms.Resize(250),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    # load the dataset
    train_dataset = PF16FaceDataset(csv_file, face_dir, factor,
                                    sample_size_per_class,
                                    class_target=True,
                                    gender_filter=None,
                                    transform=train_transform)
    test_dataset = PF16FaceDataset(csv_file, face_dir, factor,
                                   sample_size_per_class,
                                   class_target=True,
                                   gender_filter=None,
                                   transform=test_transform)

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

def train(backbone, classifier, criterion, device, train_loader, optimizer,
          epoch, writer):
    backbone.train()
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        features = backbone(data)
        output = classifier(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar('data/training-loss', loss,
                          (epoch-1)*int(len(train_loader.sampler.indices)/100)+batch_idx+1)
        if batch_idx % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.sampler.indices),
                100.*batch_idx*len(data)/len(train_loader.sampler.indices),
                loss.item()))

def test(backbone, classifier, criterion, device, test_loader, epoch, writer):
    backbone.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    all_pred = []
    all_true = []
    all_p = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features = backbone(data)
            output = classifier(features)
            # sum up batch loss
            part_loss = criterion(output, target).item()
            test_loss += part_loss * data.size(0)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.long().eq(target).sum().item()
            all_pred.append(pred.cpu().data.numpy())
            all_true.append(target.cpu().data.numpy())
            p = torch.softmax(output, dim=1)
            all_p.append(p.cpu().data.numpy()[:, 0])
    
    # plot model parameter hist
    for name, param in classifier.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    writer.add_histogram('output-p', np.concatenate(tuple(all_p)), epoch)
    #params = classifier.state_dict()
    #print(params.keys())
    #x = vutils.make_grid(params['base_model.0.weight'].clone().cpu().data,
    #                     normalize=True, scale_each=True)
    #writer.add_image('Image', x, epoch)

    test_loss /= len(test_loader.sampler.indices)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(test_loader.sampler.indices),
        100.*correct/len(test_loader.sampler.indices)))
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred))
    print(cm*1.0 / cm.sum(axis=1, keepdims=True))
    print('\n')
    writer.add_scalar('data/test-accuracy', 100.*correct/len(test_loader.sampler.indices), epoch)

    return 100.*correct/len(test_loader.sampler.indices)

def train_ensemble_model_sugar(factor, random_seed):
    """Main function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # load data for cross-validation
    data_dir = '/home/huanglj/proj'
    sample_size_per_class = 1500
    test_ratio = 0.2
    c1_sample_idx = np.arange(sample_size_per_class).tolist()
    c2_sample_idx = np.arange(sample_size_per_class).tolist()
    split_idx = int(np.floor(sample_size_per_class * test_ratio))
    # get training- and testing-samples
    print('Random seed is %s'%(random_seed))
    np.random.seed(random_seed)
    np.random.shuffle(c1_sample_idx)
    np.random.shuffle(c2_sample_idx)
    # CV
    for fold in range(int(1/test_ratio)):
        print('Fold %s/%s'%(fold+1, int(1/test_ratio)))
        train_sampler = SubsetRandomSampler(c1_sample_idx[split_idx:]+[i+sample_size_per_class for i in c2_sample_idx[split_idx:]])
        val_sampler = SubsetRandomSampler(c1_sample_idx[:split_idx]+[i+sample_size_per_class for i in c2_sample_idx[:split_idx]])
        c1_sample_idx = c1_sample_idx[split_idx:] + c1_sample_idx[:split_idx]
        c2_sample_idx = c2_sample_idx[split_idx:] + c2_sample_idx[:split_idx]
        # load data    
        train_loader, val_loader = load_data(factor,
                                             data_dir,
                                             sample_size_per_class,
                                             train_sampler,
                                             val_sampler,
                                             batch_size=100,
                                             num_workers=16,
                                             pin_memory=True)
        # model training and eval
        model_backbone = ShuffleNetV2(n_class=512, input_size=224,
                                      width_mult=1.0)
        backbone_file = './shufflefacenet_512/Backbone_shufflenet_v2_x1_0_Epoch_22_checkpoint.pth'
        model_backbone.load_state_dict(torch.load(backbone_file,
                                    map_location=lambda storage, loc: storage))
        model_backbone = model_backbone.to(device)
        classifier = clsNet1(2).to(device)
        #classifier = clsNet2(2).to(device)

        # summary writer config
        writer = SummaryWriter()
        optimizer = optim.SGD([
                        {'params': model_backbone.parameters(),
                         'weight_decay': 1e-8},
                        {'params': classifier.parameters(),
                         'weight_decay': 1e-5},
                        ],
                        lr=0.0005,
                        momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        max_patience = 15
        patience_count = 0
        max_acc = 0
        test_acc = []
        max_epoch = 50
        for epoch in range(1, max_epoch+1):
            scheduler.step()
            train(model_backbone, classifier, criterion, device, train_loader,
                  optimizer, epoch, writer)
            acc = test(model_backbone, classifier, criterion, device,
                       val_loader, epoch, writer)
            test_acc.append(acc)
            if acc >= (max_acc-0.5):
                patience_count = 0
                if acc>=max_acc:
                    max_acc = acc
                sel_epoch = epoch
                best_backbone = copy.deepcopy(model_backbone)
                best_classifier = copy.deepcopy(classifier)
            else:
                patience_count += 1
            # save model
            if patience_count==max_patience or epoch==max_epoch:
                saved_backbone_file = 'finetuned_shufflenet4%s_backbone_f%se%s.pth'%(factor.lower(), fold, sel_epoch)
                torch.save(best_backbone.state_dict(), saved_backbone_file)
                saved_clfier_file = 'finetuned_shufflenet4%s_clfier_f%se%s.pth'%(factor.lower(), fold, sel_epoch)
                torch.save(best_classifier.state_dict(), saved_clfier_file)
                # save test accruacy
                with open('test_acc.csv', 'a+') as f:
                    f.write(','.join([str(item) for item in test_acc])+'\n')
                break
 
        #writer.export_scalars_to_json('./all_scalars_%s.json'%(random_seed))
        writer.close()

def train_ensemble_model():
    """Main function."""
    # weight decay parameters
    # L: backbone = 1e-8, classifier = 1e-5
    # others: backbone = 5e-6, classifier = 5e-5, lr = 0.0005
    factor_list = ['X4']
    #factor_list = ['B', 'C', 'E', 'F', 'G', 'H', 'I',
    #               'M', 'N', 'O',
    #               'Q1', 'Q2', 'Q3', 'Q4',
    #               'X1', 'X2', 'X3', 'X4']
    seed = 10
    for f in factor_list:
        print('Factor %s'%(f))
        train_ensemble_model_sugar(f, seed)
        # rename log files
        os.system(' '.join(['mv', 'test_acc.csv',
                    'finetuned_shufflefacenet_acc_%s_1500_cv.csv'%(f.lower())]))
        os.system(' '.join(['mv', 'runs',
                    'finetuned_shufflefacenet_log_%s_1500_cv'%(f.lower())]))
    

if __name__=='__main__':
    train_ensemble_model()

