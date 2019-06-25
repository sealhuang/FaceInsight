# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import absolute_import
from __future__ import print_function

import os
import copy
import random

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from bnuclfdataset import PF16FaceDataset
import resnet as ResNet

class ExtraTransform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img, dtype=np.uint8)
        assert len(img.shape)==3
        img = img[:, :, ::-1]
        img = img.astype(np.float32)
        img -= np.array([91.4953, 103.8827, 131.0912])
        #img -= np.array([129.03, 125.7, 132.09])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def __repr__(self):
        return self.__class__.__name__


class clsNet1(nn.Module):
    
    def __init__(self, base_model, class_num):
        super(clsNet1, self).__init__()
        self.base_model = base_model
        #self.fc1 = nn.Linear(2048, 1024, bias=False)
        self.fc1 = nn.Linear(2048, 256, bias=False)
        self.drop1 = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(1024, 256, bias=False)
        self.fc2 = nn.Linear(256, 2, bias=True)
        #self.fc3 = nn.Linear(256, class_num, bias=True)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc2(x)
        return x

def load_weight_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model,
               assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

def load_model(model_weight_file):
    """Load resnet50 model as backbone."""
    model = ResNet.resnet50(num_classes=8631, include_top=False)
    #model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
    #                        bias=False)
    load_weight_dict(model, model_weight_file)
    #model.conv1.bias.data.fill_(1)
    #fc_in_dims = model.fc.in_features
    #model.fc = nn.Linear(fc_in_dims, 64, bias=False)

    return model

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
    #csv_file = os.path.join(data_dir, 'mbti_factors.csv')
    csv_file = os.path.join(data_dir, 'sel_16pf_factors.csv')
    face_dir = os.path.join(data_dir, 'bnu_aligned_faces')

    # get image stats
    #m, s = get_img_stats(csv_file, face_dir, batch_size=batch_size, 
    #                     num_workers=num_workers, pin_memory=pin_memory)
    
    # define transforms
    #normalize = transforms.Normalize(mean=[0.518, 0.493, 0.506],
    #                                 std=[0.270, 0.254, 0.277])
    #normalize = transforms.Normalize(mean=(0.507395516207, ),
    #                                 std=(0.255128989415, ))
    #transforms.RandomResizedCrop(224, scale=(0.7, 0.9), ratio=(1.0, 1.0)),
    train_transform = transforms.Compose([transforms.Resize(250),
                                          transforms.RandomCrop(224),
                                          #transforms.RandomGrayscale(p=0.2),
                                          transforms.ColorJitter(brightness=.05,
                                                                saturation=.05),
                                          transforms.RandomHorizontalFlip(),
                                          ExtraTransform()])
    test_transform = transforms.Compose([transforms.Resize(250),
                                         transforms.CenterCrop(224),
                                         ExtraTransform()])

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
    #train_dataset = MBTIFaceDataset(csv_file, face_dir, 'JP',
    #                                sample_size_per_class,
    #                                class_target=True,
    #                                gender_filter='female',
    #                                transform=train_transform)
    #test_dataset = MBTIFaceDataset(csv_file, face_dir, 'JP',
    #                               sample_size_per_class,
    #                               class_target=True,
    #                               gender_filter='female',
    #                               transform=train_transform)
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

def train(model, criterion, device, train_loader, optimizer, epoch, writer):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar('data/training-loss', loss,
                          (epoch-1)*int(len(train_loader.sampler.indices)/50)+batch_idx+1)
        if batch_idx % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.sampler.indices),
                100.*batch_idx*len(data)/len(train_loader.sampler.indices),
                loss.item()))

def test(model, criterion, device, test_loader, epoch, writer):
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
            part_loss = criterion(output, target).item()
            test_loss += part_loss * data.size(0)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            all_pred.append(pred.cpu().data.numpy())
            all_true.append(target.cpu().data.numpy())
    
    # plot model parameter hist
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    params = model.state_dict()
    #print(params.keys())
    x = vutils.make_grid(params['base_model.conv1.weight'].clone().cpu().data,
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

def train_ensemble_model_sugar(factor, random_seed):
    """Main function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
                                             batch_size=50,
                                             num_workers=15,
                                             pin_memory=True)

        # model training and eval
        model_weight_file = './resnet50_ft_weight.pkl'
        base_model = load_model(model_weight_file)
        model = clsNet1(base_model, 2).to(device)
        #print(model)

        params_update = []
        print('updated parameters:')
        for name, param in model.named_parameters():
            if name.startswith('fc'):
                params_update.append(param)
                print('\t%s'%(name))
            else:
                param.requires_grad = False

        # summary writer config
        writer = SummaryWriter()
        #writer.add_graph(model, torch.zeros(1, 3, 224, 224).to(device), False)
        optimizer = optim.SGD([
                            #{'params': params_update, 'weight_decay': 1e-8},
                                {'params': model.fc1.parameters(), 'lr': 1e-3},
                                {'params': model.fc2.parameters()}],
                            lr=1e-3, momentum=0.9, weight_decay=1e-8)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                              gamma=0.5)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        max_patience = 15
        patience_count = 0
        max_acc = 0
        test_acc = []
        max_epoch = 100
        for epoch in range(1, max_epoch+1):
            scheduler.step()
            train(model, criterion, device, train_loader, optimizer, epoch,
                  writer)
            acc = test(model, criterion, device, val_loader, epoch, writer)
            test_acc.append(acc)
            if acc >= (max_acc-0.5):
                if acc>=max_acc:
                    max_acc = acc
                patience_count = 0
                sel_epoch = epoch
                best_model = copy.deepcopy(model)
            else:
                patience_count += 1
            # save model
            if patience_count==max_patience or epoch==max_epoch:
                saved_model_file = 'finetuned_resnet4%s_f%se%s.pth'%(factor.lower(), fold, sel_epoch)
                torch.save(best_model.state_dict(), saved_model_file)
                # save test accruacy
                with open('test_acc.csv', 'a+') as f:
                    f.write(','.join([str(item) for item in test_acc])+'\n')
                break
 
        #writer.export_scalars_to_json('./all_scalars_%s.json'%(random_seed))
        writer.close()

def train_ensemble_model():
    """Main function."""
    # params
    # A. base_model: weight_decay=1e-8, classifier: weight decay=5e-8,
    #    lr=0.001, gamma=0.1
    # H. base_model: weight_decay=1e-8, classifier: weight decay=5e-8,
    #    lr=0.001, gamma=0.1
    # L. base_model: weight_decay 1e-8, classifier: weight decay 5e-8,
    #    lr=0.001, gamma=0.1
    # N. base_model: weight_decay 1e-8, classifier: weight decay 1e-8,
    #    lr=0.001, gamma=0.1
    # Q3. base_model: weight_decay 1e-8, classifier: weight decay 1e-8,
    #    lr=0.001, gamma=0.1
    # X2. base_model: weight_decay=1e-8, classifier: weight decay=1e-7,
    #    lr=0.001, gamma=0.2
    # X3. base_model: weight_decay=1e-8, classifier: weight decay=5e-7,
    #    lr=0.001, gamma=0.1
    # X4. base_model: weight_decay=1e-8, classifier: weight decay=5e-8,
    #    lr=0.001, gamma=0.1
    # F. base_model: weight_decay=1e-8, classifier: weight decay=1e-8,
    #    lr=0.001, gamma=0.1
    #factor_list = ['E', 'I', 'M', 'Q2', 'X1', 'Y1', 'Y2', 'Y3']
    factor_list = ['A']
    seed = 10
    for f in factor_list:
        print('Factor %s'%(f))
        train_ensemble_model_sugar(f, seed)
        # rename log files
        os.system(' '.join(['mv', 'test_acc.csv',
                            'finetuned_acc_%s_1500_cv.csv'%(f.lower())]))
        os.system(' '.join(['mv', 'runs',
                            'finetuned_log_%s_1500_cv'%(f.lower())]))
    

if __name__=='__main__':
    train_ensemble_model()

