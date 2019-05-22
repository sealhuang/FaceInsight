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
from bnuclfdataset import PF16FaceDataset
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from faceevolve.backbone import model_vgg_face


class clsNet1(nn.Module):
    def __init__(self, base_model, class_num):
        super(clsNet1, self).__init__()

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Linear(4096, class_num)

    def forward(self, x):
        """Pass the input tensor through each of our operations."""
        x = self.base_model(x)
        x = self.classifier(x)
        #return F.log_softmax(x, dim=1)
        return x


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
    normalize = transforms.Normalize(mean=[0.518, 0.493, 0.506],
                                     std=[0.270, 0.254, 0.277])
    #normalize = transforms.Normalize(mean=(0.507395516207, ),
    #                                 std=(0.255128989415, ))
    #transforms.RandomResizedCrop(224, scale=(0.7, 0.9), ratio=(1.0, 1.0)),
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
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        #loss = F.nll_loss(output, target)
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
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            part_loss = criterion(output, target).item()
            test_loss += part_loss * data.size(0)
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
    #print(params.keys())
    x = vutils.make_grid(params['base_model.0.weight'].clone().cpu().data,
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

def run_model_cv(factor, random_seed):
    """Main function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data for cross-validation
    data_dir = '/home/huanglj/proj'
    sample_size_per_class = 1500
    test_ratio = 0.2
    c1_sample_idx = range(sample_size_per_class)
    c2_sample_idx = range(sample_size_per_class)
    split_idx = int(np.floor(sample_size_per_class * test_ratio))
    # get training- and testing-samples
    print('Random seed is %s'%(random_seed))
    np.random.seed(random_seed)
    np.random.shuffle(c1_sample_idx)
    np.random.shuffle(c2_sample_idx)
    # CV
    for fold in range(int(1/test_ratio)):
    #for fold in range(1):
        print('Fold %s/%s'%(fold+1, int(1/test_ratio)))
        train_sampler = SubsetRandomSampler(c1_sample_idx[split_idx:]+[i+sample_size_per_class for i in c2_sample_idx[split_idx:]])
        test_sampler = SubsetRandomSampler(c1_sample_idx[:split_idx]+[i+sample_size_per_class for i in c2_sample_idx[:split_idx]])
        c1_sample_idx = c1_sample_idx[split_idx:] + c1_sample_idx[:split_idx]
        c2_sample_idx = c2_sample_idx[split_idx:] + c2_sample_idx[:split_idx]
        # load data    
        train_loader, test_loader = load_data(factor,
                                              data_dir,
                                              sample_size_per_class,
                                              train_sampler,
                                              test_sampler,
                                              batch_size=50,
                                              num_workers=25,
                                              pin_memory=True)
        # model training and eval
        model_backbone = model_vgg_face.VGG_Face_torch
        backbone_weights = os.path.join(
                                os.path.dirname(model_vgg_face.__file__),
                                'model_vgg_face.pth')
        model_backbone.load_state_dict(torch.load(backbone_weights))
        model = clsNet1(model_backbone, 2).to(device)
        #print(model.base_model)
        # grad config
        #for para in list(model.base_model.parameters()):
        #    para.requires_grad = False

        # summary writer config
        writer = SummaryWriter()
        optimizer = optim.SGD([
                        {'params': model.base_model.parameters(),
                         'weight_decay': 1e-8},
                        {'params': model.classifier.parameters(),
                         'weight_decay': 5e-8}
                        ], lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)
        #optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        test_acc = []
        for epoch in range(1, 31):
            scheduler.step()
            train(model, criterion, device, train_loader, optimizer, epoch,
                  writer)
            acc = test(model, criterion, device, test_loader, epoch, writer)
            test_acc.append(acc)

        # save test accruacy
        with open('test_acc.csv', 'a+') as f:
            f.write(','.join([str(item) for item in test_acc])+'\n')
    
        #writer.export_scalars_to_json('./all_scalars_%s.json'%(random_seed))
        writer.close()

def run_model(factor, random_seed):
    """Main function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data for model training
    data_dir = '/home/huanglj/proj'
    sample_size_per_class = 1500
    test_ratio = 0.1
    c1_sample_idx = range(sample_size_per_class)
    c2_sample_idx = range(sample_size_per_class)
    split_idx = int(np.floor(sample_size_per_class * test_ratio))
    # get training- and validation-samples
    print('Random seed is %s'%(random_seed))
    np.random.seed(random_seed)
    np.random.shuffle(c1_sample_idx)
    np.random.shuffle(c2_sample_idx)
    # split data
    train_sampler = SubsetRandomSampler(c1_sample_idx[split_idx:]+[i+sample_size_per_class for i in c2_sample_idx[split_idx:]])
    val_sampler = SubsetRandomSampler(c1_sample_idx[:split_idx]+[i+sample_size_per_class for i in c2_sample_idx[:split_idx]])
    # load data    
    train_loader, val_loader = load_data(factor,
                                         data_dir,
                                         sample_size_per_class,
                                         train_sampler,
                                         val_sampler,
                                         batch_size=50,
                                         num_workers=25,
                                         pin_memory=True)
    # model training and eval
    model_backbone = model_vgg_face.VGG_Face_torch
    backbone_weights = os.path.join(os.path.dirname(model_vgg_face.__file__),
                                    'model_vgg_face.pth')
    model_backbone.load_state_dict(torch.load(backbone_weights))
    model = clsNet1(model_backbone, 2).to(device)
    #print(model.base_model)
    # grad config
    #for para in list(model.base_model.parameters()):
    #    para.requires_grad = False

    # summary writer config
    writer = SummaryWriter()
    optimizer = optim.SGD([{'params': model.base_model.parameters(),
                            'weight_decay': 1e-8},
                           {'params': model.classifier.parameters(),
                             'weight_decay': 5e-8}],
                          lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    max_patience = 5
    patience_conut = 0
    max_acc = 0
    test_acc = []
    for epoch in range(1, 31):
        scheduler.step()
        train(model, criterion, device, train_loader, optimizer, epoch, writer)
        acc = test(model, criterion, device, test_loader, epoch, writer)
        test_acc.append(acc)
        if test_acc > max_acc:
            max_acc = test_acc
            patience_count = 0
        else:
            patience_count += 1
        # save model
        if patience_count==max_patience or epoch==30:
            saved_model_file = 'finetuned_vggface_model4%s.pth'%(factor.lower())
            torch.save(model.state_dict(), saved_model_file)
            # save test accruacy
            with open('test_acc.csv', 'a+') as f:
                f.write(','.join([str(item) for item in test_acc])+'\n')
    
    #writer.export_scalars_to_json('./all_scalars_%s.json'%(random_seed))
    writer.close()

def cv_main():
    """Main function."""
    #seeds = [10, 25, 69, 30, 22, 91, 65, 83, 11, 8]
    factor_list = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N',
                   'O', 'Q1', 'Q2', 'Q3', 'Q4', 'X1', 'X2', 'X3', 'X4',
                   'Y1', 'Y2', 'Y3', 'Y4']
    seeds = [10]
    for i in seeds:
        for f in factor_list:
            print('Factor %s'%(f))
            run_model_cv(f, i)
            # rename log files
            os.system(' '.join(['mv', 'test_acc.csv',
                                'fine_tune_%s_1500_celoss.csv'%(f.lower())]))
            os.system(' '.join(['mv', 'runs',
                                'fine_tune_%s_1500_combine'%(f.lower())]))
    
def main():
    """Main function."""
    #seeds = [10, 25, 69, 30, 22, 91, 65, 83, 11, 8]
    factor_list = ['A', 'E', 'F', 'H', 'I', 'L', 'M', 'N',
                   'Q2', 'Q3', 'X1', 'X2', 'X3', 'X4',
                   'Y1', 'Y2', 'Y3']
    seeds = [10]
    for i in seeds:
        for f in factor_list:
            print('Factor %s'%(f))
            run_model(f, i)
            # rename log files
            os.system(' '.join(['mv', 'test_acc.csv',
                                'fine_tune_%s_1350.csv'%(f.lower())]))
            os.system(' '.join(['mv', 'runs',
                                'fine_tune_%s_1350'%(f.lower())]))
    

if __name__=='__main__':
    #cv_main()
    main()


