# vi: set ft=python sts=4 ts=4 sw=4 et:

import torch


configurations = {
    1: dict(
        # random seed for reproduce results
        SEED = 1337,

        # the parent root where your train/val/test data are stored
        DATA_ROOT = '/media/pc/6T/jasonjzhao/data/faces_emore',
        # the root to buffer your checkpoints
        MODEL_ROOT = '/media/pc/6T/jasonjzhao/buffer/model',
        # the root to log your train/val status
        LOG_ROOT = '/media/pc/6T/jasonjzhao/buffer/log',
        # the root to resume training from a saved checkpoint
        BACKBONE_RESUME_ROOT = './', 
        # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './',

        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152',
        #           'IR_50', 'IR_101', 'IR_152',
        #           'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        BACKBONE_NAME = 'IR_SE_50',
        # support: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        HEAD_NAME = 'ArcFace',
        # support: ['Focal', 'Softmax']
        LOSS_NAME = 'Focal',

        # support: [112, 112] and [224, 224]
        INPUT_SIZE = [112, 112],
        # for normalize inputs to [-1, 1]
        RGB_MEAN = [0.5, 0.5, 0.5],
        RGB_STD = [0.5, 0.5, 0.5],
        # feature dimension
        EMBEDDING_SIZE = 512,
        BATCH_SIZE = 512,
        # whether drop the last batch to ensure consistent batch_norm statistics
        DROP_LAST = True,
        # initial LR
        LR = 0.1,
        # total epoch number (use the firt 1/25 epochs to warm up)
        NUM_EPOCH = 125,
        # do not apply to batch_norm parameters
        WEIGHT_DECAY = 5e-4,
        MOMENTUM = 0.9,
        # epoch stages to decay learning rate
        STAGES = [35, 65, 95],

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # flag to use multiple GPUs; if you choose to train with single GPU,
        # you should first run "export CUDA_VISILE_DEVICES=device_id" to specify
        # the GPU card you want to use
        MULTI_GPU = True,
        # specify your GPU ids
        GPU_ID = [0, 1, 2, 3],
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
),
}

