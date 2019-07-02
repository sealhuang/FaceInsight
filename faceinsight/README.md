# High-Performance Face Recognition Library based on PyTorch

## Contents
* [Introduction](#Introduction)
* [Pre-Requisites](#Pre-Requisites)

****
### Introduction 

<img src="https://github.com/sealhuang/FaceInsight/blob/master/faceinsight/disp/Fig1.png" width="450px"/>  <img src="https://github.com/sealhuang/FaceInsight/blob/master/faceinsight/disp/Fig17.png" width="400px"/>

* This repo provides a comprehensive face recognition library for face related analytics \& applications, including face alignment (detection, landmark localization, affine transformation, *etc.*), data processing (*e.g.*, augmentation, data balancing, normalization, *etc.*), various backbones (*e.g.*, [ResNet](https://arxiv.org/pdf/1512.03385.pdf), [IR](https://arxiv.org/pdf/1512.03385.pdf), [IR-SE](https://arxiv.org/pdf/1709.01507.pdf), ResNeXt, SE-ResNeXt, DenseNet, [LightCNN](https://arxiv.org/pdf/1511.02683.pdf), MobileNet, ShuffleNet, DPN, *etc.*), various losses (*e.g.*, Softmax, [Focal](https://arxiv.org/pdf/1708.02002.pdf), Center, [SphereFace](https://arxiv.org/pdf/1704.08063.pdf), [CosFace](https://arxiv.org/pdf/1801.09414.pdf), [AmSoftmax](https://arxiv.org/pdf/1801.05599.pdf), [ArcFace](https://arxiv.org/pdf/1801.07698.pdf), Triplet, *etc.*) and bags of tricks for improving performance (*e.g.*, training refinements, model tweaks, knowledge distillation, *etc.*).
* The current distributed training schema with multi-GPUs under PyTorch and other mainstream platforms parallels the backbone across multi-GPUs while relying on a single master to compute the final bottleneck (fully-connected/softmax) layer. This is not an issue for conventional face recognition with moderate number of identities. However, it struggles with large-scale face recognition, which requires recognizing millions of identities in the real world. The master can hardly hold the oversized final layer while the slaves still have redundant computation resource, leading to small-batch training or even failed training. To address this problem, this repo provides a highly-elegant, effective and efficient distributed training schema with multi-GPUs under PyTorch, supporting not only the backbone, but also the head with the fully-connected (softmax) layer, to facilitate high-performance large-scale face recognition.
* All data before \& after alignment, source codes and trained models are provided.
* This repo can help researchers/engineers develop high-performance deep face recognition models and algorithms quickly for practical use and deployment.

****
### Pre-Requisites 

* Linux or macOS
* Python 3.7
* PyTorch > 1.0 (for traininig \& validation, install w/ `pip install torch torchvision`)
* TensorFlow > 1.12 (optinal, for visualization, install w/ `pip install tensorflow-gpu`)
* OpenCV 3.4.5 (install w/ `pip install opencv-python`)
* bcolz 1.2.0 (install w/ `pip install bcolz`)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU. We used 4-8 NVIDIA Tesla P40 in parallel.

