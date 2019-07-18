# vi: set ft=python sts=4 ts=4 sw=4 et:

import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from faceinsight.models.shufflenet_v2 import ShuffleNetV2
from faceinsight.models.mobilefacenet import MobileFaceNet
from faceinsight.head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from faceinsight.loss.focal import FocalLoss
from faceinsight.util.utils import get_time
from faceinsight.util.utils import make_weights_for_balanced_classes 
from faceinsight.util.utils import schedule_lr, warm_up_lr
from faceinsight.util.meter import AverageMeter, accuracy
from faceinsight.io.pubdataloader import get_lfw_val_pair
from faceinsight.util.utils import perform_lfw_val, buffer_val

from config import configurations


if __name__ == '__main__':

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    # random seed for reproduce results
    SEED = cfg['SEED']
    torch.manual_seed(SEED)
 
    # the parent root where your train/val/test data are stored
    DATA_ROOT = cfg['DATA_ROOT']
    # the root to buffer your checkpoints
    MODEL_ROOT = cfg['MODEL_ROOT']
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT, mode=0o755)
    # the root to log your train/val status
    LOG_ROOT = cfg['LOG_ROOT']
    if not os.path.exists(LOG_ROOT):
        os.makedirs(LOG_ROOT, mode=0o755)
    # the root to resume training from a saved checkpoint
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']
    # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']

    BACKBONE_NAME = cfg['BACKBONE_NAME'] 
    HEAD_NAME = cfg['HEAD_NAME']
    LOSS_NAME = cfg['LOSS_NAME']

    INPUT_SIZE = cfg['INPUT_SIZE']
    # for normalize inputs
    RGB_MEAN = cfg['RGB_MEAN']
    RGB_STD = cfg['RGB_STD']
    # feature dimension
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']
    BATCH_SIZE = cfg['BATCH_SIZE']
    # whether drop the last batch to ensure consistent batch_norm statistics
    DROP_LAST = cfg['DROP_LAST']
    # initial LR
    LR = cfg['LR']
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    # epoch stages to decay learning rate
    STAGES = cfg['STAGES']

    DEVICE = cfg['DEVICE']
    # flag to use multiple GPUs
    MULTI_GPU = cfg['MULTI_GPU']
    # specify your GPU ids
    GPU_ID = cfg['GPU_ID']
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print('=' * 60)
    print('Overall Configurations:')
    print(cfg)
    print('=' * 60)

    # writer for buffering intermedium results
    writer = SummaryWriter(LOG_ROOT)

    train_transform = transforms.Compose([
                        transforms.Resize(250),
                        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
                        ])
    train_data_dir = os.path.join(DATA_ROOT, 'CASIA-WebFace', 'cropped')
    dataset_train = datasets.ImageFolder(train_data_dir, train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs,
                                                len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights),
                                                     replacement=False)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=BATCH_SIZE,
                                               sampler=sampler,
                                               pin_memory=PIN_MEMORY,
                                               num_workers=NUM_WORKERS,
                                               drop_last=DROP_LAST)

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    # get val data
    lfw_img_dir = os.path.join(DATA_ROOT, 'lfw', 'cropped')
    lfw_pair_file = os.path.join(DATA_ROOT, 'lfw', 'pairs.txt')
    lfw_pairs, lfw_issame = get_lfw_val_pair(lfw_pair_file, lfw_img_dir)

    # ======= model & loss & optimizer =======#
    if BACKBONE_NAME=='shufflenet_v2_x0_5':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=224,
                                width_mult=0.5)
    elif BACKBONE_NAME=='shufflenet_v2_x1_0':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=224,
                                width_mult=1.0)
    elif BACKBONE_NAME=='shufflenet_v2_x1_5':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=224,
                                width_mult=1.5)
    elif BACKBONE_NAME=='shufflenet_v2_x2_0':
        BACKBONE = ShuffleNetV2(n_class=EMBEDDING_SIZE,
                                input_size=224,
                                width_mult=2.0)
    elif BACKBONE_NAME=='mobilefacenet':
        BACKBONE = MobileFaceNet(EMBEDDING_SIZE)
    else:
        pass

    print('=' * 60)
    print(BACKBONE)
    print('{} Backbone Generated'.format(BACKBONE_NAME))
    print('=' * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE,
                                    out_features=NUM_CLASS,
                                    device_id=GPU_ID),
                 'CosFace': CosFace(in_features=EMBEDDING_SIZE,
                                    out_features=NUM_CLASS,
                                    device_id=GPU_ID),
                 'SphereFace': SphereFace(in_features=EMBEDDING_SIZE,
                                          out_features=NUM_CLASS,
                                          device_id=GPU_ID),
                 'Am_softmax': Am_softmax(in_features=EMBEDDING_SIZE,
                                          out_features=NUM_CLASS,
                                          device_id=GPU_ID),
                }
    HEAD = HEAD_DICT[HEAD_NAME]
    print('=' * 60)
    print(HEAD)
    print('{} Head Generated'.format(HEAD_NAME))
    print('=' * 60)

    LOSS_DICT = {'Focal': FocalLoss(), 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    print('=' * 60)
    print(LOSS)
    print('{} Loss Generated'.format(LOSS_NAME))
    print('=' * 60)

    # define optimizer
    # separate batch_norm parameters from others; do not do weight decay for
    # batch_norm parameters to improve the generalizability
    # For ShuffleNet
    #ignored_params = list(map(id, BACKBONE.classifier.parameters()))
    # For MobileFaceNet
    ignored_params = list(map(id, BACKBONE.linear.parameters()))
    ignored_params += list(map(id, HEAD.weight))
    backbone_params_only_bn = []
    backbone_params_prelu = []
    for m in BACKBONE.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            ignored_params += list(map(id, m.parameters()))
            backbone_params_only_bn += m.parameters()
        elif isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            backbone_params_prelu += m.parameters()
        else:
            pass
    backbone_params_base = filter(lambda p: id(p) not in ignored_params,
                                  BACKBONE.parameters())
    OPTIMIZER = optim.SGD([{'params': backbone_params_base,
                            'weight_decay': WEIGHT_DECAY},
                           {'params': backbone_params_only_bn,
                            'weight_decay': 0.0},
                           #{'params': BACKBONE.classifier.parameters(),
                           # 'weight_decay': WEIGHT_DECAY*1e-1},
                           {'params': backbone_params_prelu,
                            'weight_decay': 0.0},
                           {'params': BACKBONE.linear.parameters(),
                            'weight_decay': WEIGHT_DECAY*1e1},
                           {'params': HEAD.weight,
                            'weight_decay': WEIGHT_DECAY*1e1},
                          ],
                          lr=LR, momentum=MOMENTUM, nesterov=False)
    print('=' * 60)
    print(OPTIMIZER)
    print('Optimizer Generated')
    print('=' * 60)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print('=' * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and \
           os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print('No Checkpoint Found!')
        print('=' * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
    torch.backends.cudnn.benchmark = True

    # ======= train & validation & save checkpoint =======#
    # frequency to display training loss & acc
    DISP_FREQ = len(train_loader) // 100  

    # use the first 1/25 epochs to warm up
    #NUM_EPOCH_WARM_UP = NUM_EPOCH // 25
    NUM_EPOCH_WARM_UP = 0
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP

    # start training
    batch = 0
    for epoch in range(NUM_EPOCH):
       # adjust LR for each training stage after warm up, you can also
        # choose to adjust LR manually (with slight modification) once
        # plaueau observed
        for stage_thresh in STAGES:
            if epoch+1==stage_thresh:
                schedule_lr(OPTIMIZER)
                break

        # set to training mode
        BACKBONE.train()
        HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for inputs, labels in tqdm(iter(train_loader)):
            # adjust LR for each training batch during warm up
            if batch+1 <= NUM_BATCH_WARM_UP:
                warm_up_lr(batch+1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # dispaly training loss & acc every DISP_FREQ
            if ((batch+1) % DISP_FREQ == 0) and batch != 0:
                print('=' * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch+1, NUM_EPOCH, batch+1, len(train_loader)*NUM_EPOCH,
                    loss=losses, top1=top1, top5=top5))
                print('=' * 60)

            batch += 1

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar('Training_Loss', epoch_loss, epoch+1)
        writer.add_scalar('Training_Accuracy', epoch_acc, epoch+1)
        print('=' * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch+1, NUM_EPOCH, loss=losses, top1=top1, top5=top5))
        print('=' * 60)

        ## plot model parameter hist
        for name, param in BACKBONE.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(),epoch+1)
        for name, param in HEAD.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(),epoch+1)
        bb_params = BACKBONE.state_dict()
        # for shufflenet
        #x = vutils.make_grid(bb_params['conv1.0.weight'].clone().cpu().data,
        #                     normalize=True, scale_each=True)
        #writer.add_image('conv1', x, epoch+1)
        #x = vutils.make_grid(bb_params['global_weight.conv.weight'].clone().cpu().data,
        #                     normalize=True, scale_each=True)
        #writer.add_image('global_weight', x, epoch+1)
        # for mobilefacenet
        x = vutils.make_grid(bb_params['conv1.conv.weight'].clone().cpu().data,
                             normalize=True, scale_each=True)
        writer.add_image('conv1', x, epoch+1)
        x = vutils.make_grid(bb_params['conv_6_dw.conv.weight'].clone().cpu().data,
                             normalize=True, scale_each=True)
        writer.add_image('global_weight', x, epoch+1)

        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        print('=' * 60)
        print('Perform Evaluation on LFW, and Save Checkpoints...')
        # Val score for LFW
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_lfw_val(
                            MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, lfw_pairs, lfw_issame)
        buffer_val(writer, 'LFW', accuracy_lfw, best_threshold_lfw,
                   roc_curve_lfw, batch+1)
        print('Epoch {}/{}, Evaluation: LFW Acc: {}'.format(epoch+1, NUM_EPOCH,
                                                            accuracy_lfw))
        print('=' * 60)

        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, 'Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth'.format(BACKBONE_NAME, epoch+1, batch, get_time())))
            torch.save(HEAD.module.state_dict(), os.path.join(MODEL_ROOT, 'Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth'.format(HEAD_NAME, epoch+1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, 'Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth'.format(BACKBONE_NAME, epoch+1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, 'Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth'.format(HEAD_NAME, epoch+1, batch, get_time())))


