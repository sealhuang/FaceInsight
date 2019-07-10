# vi: set ft=python sts=4 ts=4 sw=4 et:

import torch


configurations = {
    1: dict(
        # random seed for reproduce results
        SEED = 1337,
        
        # the parent root where your train/val/test data are stored
        DATA_ROOT = '/home/huanglj/database',
        # the root to buffer your checkpoints
        MODEL_ROOT = '/home/huanglj/proj/buffer/model_facenet',
        # the root to log your train/val status
        LOG_ROOT = '/home/huanglj/proj/buffer/log_facenet', 
        # the root to resume training from a saved checkpoint
        BACKBONE_RESUME_ROOT = './',
        # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './',

        # ['shufflenet_v2_x0_5', 'shufflenet_v2_1_0', 'mobilefacenet']
        BACKBONE_NAME = 'shufflenet_v2_x1_0',
        # HEAD: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        HEAD_NAME = 'ArcFace',
        # support: ['Focal', 'Softmax']
        LOSS_NAME = 'Softmax',

        # support: [112, 112] and [224, 224]
        INPUT_SIZE = [224, 224],
        # for normalize inputs to [-1, 1]
        RGB_MEAN = [0.5, 0.5, 0.5],
        RGB_STD = [0.5, 0.5, 0.5],
        # feature dimension
        EMBEDDING_SIZE = 512,
        BATCH_SIZE = 128,
        # whether drop the last batch to ensure consistent batch_norm statistics
        DROP_LAST = True,
        # initial LR
        LR = 1e-2,
        # total epoch number (use the firt 1/25 epochs to warm up)
        NUM_EPOCH = 100,
        # do not apply to batch_norm parameters
        WEIGHT_DECAY = 1e-5,
        MOMENTUM = 0.9,
        # batch stages to decay learning rate
        STAGES = [16, 41, 61],

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # flag to use multiple GPUs; if you choose to train with single GPU,
        # you should first run "export CUDA_VISILE_DEVICES=device_id" to specify
        # the GPU card you want to use
        MULTI_GPU = False,
        # specify your GPU ids
        GPU_ID = [0],
        PIN_MEMORY = True,
        NUM_WORKERS = 16,
),
}
