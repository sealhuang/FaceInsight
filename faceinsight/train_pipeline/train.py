# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

from config import configurations
from faceinsight.models.resnet_models import ResNet_50, ResNet_101, ResNet_152
from faceinsight.models.resnet_models import separate_resnet_bn_paras
from faceinsight.models.irse_models import IR_50, IR_101, IR_152
from faceinsight.models.irse_models import IR_SE_50, IR_SE_101, IR_SE_152
from faceinsight.models.irse_models import separate_irse_bn_paras
from faceinsight.head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from faceinsight.loss.focal import FocalLoss
from faceinsight.util.utils import make_weights_for_balanced_classes, warm_up_lr, schedule_lr, get_time
from faceinsight.util.utils import get_val_pair, perform_val, buffer_val, AverageMeter, accuracy


if __name__ == '__main__':
    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    # random seed for reproduce results
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    # the parent root where your train/val/test data are stored
    DATA_ROOT = cfg['DATA_ROOT'] 
    # the root to buffer your checkpoints
    MODEL_ROOT = cfg['MODEL_ROOT']
    # the root to log your train/val status
    LOG_ROOT = cfg['LOG_ROOT']
    # the root to resume training from a saved checkpoint
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']
    # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']

    # support: ['ResNet_50', 'ResNet_101', 'ResNet_152',
    #           'IR_50', 'IR_101', 'IR_152',
    #           'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    BACKBONE_NAME = cfg['BACKBONE_NAME']
    # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    HEAD_NAME = cfg['HEAD_NAME']
    # support: ['Focal', 'Softmax']
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
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    # writer for buffering intermedium results
    writer = SummaryWriter(LOG_ROOT)

    # image preprocessing for training image
    train_transform = transforms.Compose([
                    transforms.Resize([int(128 * INPUT_SIZE[0] / 112),
                                       int(128 * INPUT_SIZE[0] / 112)]),
                    transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
                    ])

    dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, 'imgs'),
                                         train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs,
                                                len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler=torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=BATCH_SIZE,
                                               sampler=sampler,
                                               pin_memory=PIN_MEMORY,
                                               num_workers=NUM_WORKERS,
                                               drop_last=DROP_LAST)

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    # XXX: select the val database
    lfw, lfw_issame = get_val_pair(DATA_ROOT, 'lfw')
    cfp_ff, cfp_ff_issame = get_val_pair(DATA_ROOT, 'cfp_ff')
    cfp_fp, cfp_fp_issame = get_val_pair(DATA_ROOT, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(DATA_ROOT, 'agedb_30')
    calfw, calfw_issame = get_val_pair(DATA_ROOT, 'calfw')
    cplfw, cplfw_issame = get_val_pair(DATA_ROOT, 'cplfw')
    vgg2_fp, vgg2_fp_issame = get_val_pair(DATA_ROOT, 'vgg2_fp')


    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                     'ResNet_101': ResNet_101(INPUT_SIZE), 
                     'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE), 
                     'IR_101': IR_101(INPUT_SIZE), 
                     'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                     'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                     'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

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
                                          device_id=GPU_ID)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(), 
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    # separate batch_norm parameters from others; do not do weight decay
    # for batch_norm parameters to improve the generalizability
    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE)
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn,
                            'weight_decay': WEIGHT_DECAY},
                           {'params': backbone_paras_only_bn}],
                          lr=LR, momentum=MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)


    #======= train & validation & save checkpoint =======#
    # frequency to display training loss & acc
    DISP_FREQ = len(train_loader) // 100

    # use the first 1/25 epochs to warm up
    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP
    
    # start training
    batch = 0
    for epoch in range(NUM_EPOCH):
        # adjust LR for each training stage after warm up, you can also
        # choose to adjust LR manually (with slight modification) once
        # plaueau observed
        if epoch == STAGES[0]: 
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        # set to training mode
        BACKBONE.train()
        HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for inputs, labels in tqdm(iter(train_loader)):
            # adjust LR for each training batch during warm up
            # XXX: if condition complex
            if (epoch+1<=NUM_EPOCH_WARM_UP) and (batch+1<=NUM_BATCH_WARM_UP): 
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
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch+1, NUM_EPOCH, batch+1, len(train_loader)*NUM_EPOCH,
                    loss=losses, top1=top1, top5=top5))
                print("=" * 60)
 
            batch += 1 

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, NUM_EPOCH, loss=losses, top1=top1, top5=top5))
        print("=" * 60)

        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        print("=" * 60)
        print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
        # Val score for LFW
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(
                            MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, lfw, lfw_issame)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw,
                   roc_curve_lfw, epoch+1)
        # Val score for CFP_FF
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(
                            MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, cfp_ff, cfp_ff_issame)
        buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff,
                   roc_curve_cfp_ff, epoch+1)
        # Val score for CFP_FP
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(
                            MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, cfp_fp, cfp_fp_issame)
        buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp,
                   roc_curve_cfp_fp, epoch + 1)
        # Val score for AgeDB
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(
                            MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, agedb, agedb_issame)
        buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb,
                   roc_curve_agedb, epoch + 1)
        # Val score for CALFW
        accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(
                            MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, calfw, calfw_issame)
        buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw,
                   roc_curve_calfw, epoch + 1)
        # Val score for CPLFW
        accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(
                            MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, cplfw, cplfw_issame)
        buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw,
                   roc_curve_cplfw, epoch + 1)
        # Val score for VGGFace2_FP
        accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = \
                perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                            BACKBONE, vgg2_fp, vgg2_fp_issame)
        buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp,
                   best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
        print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw, accuracy_vgg2_fp))
        print("=" * 60)

        # save checkpoints per epoch
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))

