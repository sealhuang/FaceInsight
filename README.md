# FaceInsight: Extracing interesting and valuable information from Faces

## Contents
* [Introduction](#Introduction)
* [Pre-Requisites](#Pre-Requisites)

****
### Introduction 

This repo provides some face-related applications and a comprehensive face recognition library (the code in this repo is mainly inherited from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)).

* Currently, I train a DL model to learn the relation between one's face and her/his personality. Thus the model could infer one's personality based on a face image ([A demo of `facenality`](http://xinlimian.benbenedu.cn:5000)). Some researchers have shown its feasibility, *e.g.*, [article 1](https://www.nature.com/articles/s41598-020-65358-6), [article 2](https://link.springer.com/article/10.1007/s11633-017-1085-8), and [article 3](https://www.pnas.org/content/105/32/11087.short).

* The face recognition library for face related analytics \& applications, including face alignment (detection, landmark localization, affine transformation, *etc.*), data processing (*e.g.*, augmentation, data balancing, normalization, *etc.*), various backbones (*e.g.*, [ResNet](https://arxiv.org/pdf/1512.03385.pdf), [IR](https://arxiv.org/pdf/1512.03385.pdf), [IR-SE](https://arxiv.org/pdf/1709.01507.pdf), ResNeXt, SE-ResNeXt, DenseNet, [LightCNN](https://arxiv.org/pdf/1511.02683.pdf), MobileNet, ShuffleNet, DPN, *etc.*), various losses (*e.g.*, Softmax, [Focal](https://arxiv.org/pdf/1708.02002.pdf), Center, [SphereFace](https://arxiv.org/pdf/1704.08063.pdf), [CosFace](https://arxiv.org/pdf/1801.09414.pdf), [AmSoftmax](https://arxiv.org/pdf/1801.05599.pdf), [ArcFace](https://arxiv.org/pdf/1801.07698.pdf), Triplet, *etc.*) and bags of tricks for improving performance (*e.g.*, training refinements, model tweaks, knowledge distillation, *etc.*). For more information and updates, please see [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch).

****
### Pre-Requisites 

* Linux or macOS
* Python 3.7
* PyTorch > 1.0 (for traininig \& validation, install w/ `pip install torch torchvision`)
* TensorFlow > 1.12 (optinal, for visualization, install w/ `pip install tensorflow-gpu`)
* OpenCV 3.4.5 (install w/ `pip install opencv-python`)
* bcolz 1.2.0 (install w/ `pip install bcolz`)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU.

****
### Repo-structure

* The source code of `facenality` is in `faceinsight/deploy/facex`

* Folder `faceinsight/detection` holds the code of face detection and alignment based on the work of [MTCNN](https://arxiv.org/pdf/1604.02878.pdf).

* Folder `faceinsight/proj/facetraits` holds the code for tarining the personality inference model.

