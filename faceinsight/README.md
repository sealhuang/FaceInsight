# High-Performance Face Recognition Library based on PyTorch

## Contents
* [Introduction](#Introduction)
* [Pre-Requisites](#Pre-Requisites)
* [Data Zoo](#Data-Zoo)

****
### Introduction 

<img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig1.png" width="450px"/>  <img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig17.png" width="400px"/>

* This repo provides a comprehensive face recognition library for face related analytics \& applications, including face alignment (detection, landmark localization, affine transformation, *etc.*), data processing (*e.g.*, augmentation, data balancing, normalization, *etc.*), various backbones (*e.g.*, [ResNet](https://arxiv.org/pdf/1512.03385.pdf), [IR](https://arxiv.org/pdf/1512.03385.pdf), [IR-SE](https://arxiv.org/pdf/1709.01507.pdf), ResNeXt, SE-ResNeXt, DenseNet, [LightCNN](https://arxiv.org/pdf/1511.02683.pdf), MobileNet, ShuffleNet, DPN, *etc.*), various losses (*e.g.*, Softmax, [Focal](https://arxiv.org/pdf/1708.02002.pdf), Center, [SphereFace](https://arxiv.org/pdf/1704.08063.pdf), [CosFace](https://arxiv.org/pdf/1801.09414.pdf), [AmSoftmax](https://arxiv.org/pdf/1801.05599.pdf), [ArcFace](https://arxiv.org/pdf/1801.07698.pdf), Triplet, *etc.*) and bags of tricks for improving performance (*e.g.*, training refinements, model tweaks, knowledge distillation, *etc.*).
* The current distributed training schema with multi-GPUs under PyTorch and other mainstream platforms parallels the backbone across multi-GPUs while relying on a single master to compute the final bottleneck (fully-connected/softmax) layer. This is not an issue for conventional face recognition with moderate number of identities. However, it struggles with large-scale face recognition, which requires recognizing millions of identities in the real world. The master can hardly hold the oversized final layer while the slaves still have redundant computation resource, leading to small-batch training or even failed training. To address this problem, this repo provides a highly-elegant, effective and efficient distributed training schema with multi-GPUs under PyTorch, supporting not only the backbone, but also the head with the fully-connected (softmax) layer, to facilitate high-performance large-scale face recognition.
* All data before \& after alignment, source codes and trained models are provided.
* This repo can help researchers/engineers develop high-performance deep face recognition models and algorithms quickly for practical use and deployment.

****
### Pre-Requisites 

* Linux or macOS
* Python 3.7
* PyTorch 1.0 (for traininig \& validation, install w/ `pip install torch torchvision`)
* MXNet 1.3.1 (optinal, for data processing, install w/ `pip install mxnet-cu90`)
* TensorFlow 1.12 (optinal, for visualization, install w/ `pip install tensorflow-gpu`)
* OpenCV 3.4.5 (install w/ `pip install opencv-python`)
* bcolz 1.2.0 (install w/ `pip install bcolz`)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU. We used 4-8 NVIDIA Tesla P40 in parallel.

****
### Data Zoo 

|Database|Version|\#Identity|\#Image|\#Frame|\#Video|Download Link|
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Raw|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1VzSI_xqiBw-uHKyRbi6zzw)|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Align_250x250|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/11h-QIrhuszY3PzT17Q5eXw8yrewgqX7m/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1Ir8kAcQjBJA6A_pWPL9ozQ)|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Align_112x112|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1Ew5JZ266bkg00jB5ICt78g)|
|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|Raw|4,025|12,174|-|-|[Google Drive](https://drive.google.com/file/d/1LcIDIfeZ027tbyUJDbaDt12ZoMVJuoMp/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/17IzL_nGzedup1gcPuob0NQ)|
|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|Align_112x112|4,025|12,174|-|-|[Google Drive](https://drive.google.com/file/d/1kpmcDeDmPqUcI5uX0MCBzpP_8oQVojzW/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1IxqyLFfHNQaj3ibjc7Vcvg)|
|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|Raw|3,884|11,652|-|-|[Google Drive](https://drive.google.com/file/d/1WipxZ1QXs_Fi6Y5qEFDayEgos3rHDRnS/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1gJuZZcm-2crTrqKI0sa5sA)|
|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|Align_112x112|3,884|11,652|-|-|[Google Drive](https://drive.google.com/file/d/14vPvDngGzsc94pQ4nRNfuBTxdv7YVn2Q/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1uqK2LAEE91HYqllgsWcj9A)|
|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|Raw_v1|10,575|494,414|-|-|[Baidu Drive](https://pan.baidu.com/s/1xh073sKX3IYp9xPm9S6F5Q)|
|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|Raw_v2|10,575|494,414|-|-|[Google Drive](https://drive.google.com/file/d/19R6Svdj5HbUA0y6aJv3P1WkIR5wXeCnO/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1cZqsRxln-JmrA4xevLfjYQ)|
|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|Clean|10,575|455,594|-|-|[Google Drive](https://drive.google.com/file/d/1wJC2aPA4AC0rI-tAL2BFs2M8vfcpX-w6/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1x_VJlG9WV1OdrrJ7ARUZQw)|
|[MS-Celeb-1M](https://arxiv.org/pdf/1607.08221.pdf)|Clean|100,000|5,084,127|-|-|[Google Drive](https://drive.google.com/file/d/18FxgfXgKwuYzY3DmWJXNJuY51TPmC9yH/view?usp=sharing)|
|[MS-Celeb-1M](https://arxiv.org/pdf/1607.08221.pdf)|Align_112x112|85,742|5,822,653|-|-|[Google Drive](https://drive.google.com/file/d/1X202mvYe5tiXFhOx82z4rPiPogXD435i/view?usp=sharing)|
|[Vggface2](https://arxiv.org/pdf/1710.08092.pdf)|Clean|8,631|3,086,894|-|-|[Google Drive](https://drive.google.com/file/d/1jdZw6ZmB7JRK6RS6QP3YEr2sufJ5ibtO/view?usp=sharing)|
|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|Align_112x112|-|-|-|-|[Google Drive](https://drive.google.com/file/d/1N7QEEQZPJ2s5Hs34urjseFwIoPVSmn4r/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1STSgORPyRT-eyk5seUTcRA)|
|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|Raw|570|16,488|-|-|[Google Drive](https://drive.google.com/file/d/1FoZDyzTrs8r_oFM3Xqmi3iAHsnoirTRA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1-E_hkW-bXsXNYRiAhRPM7A)|
|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|Align_112x112|570|16,488|-|-|[Google Drive](https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1ehwmQ4M7WpLylV83uUBxiA)|
|[IJB-A](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Klare_Pushing_the_Frontiers_2015_CVPR_paper.pdf)|Clean|500|5,396|20,369|2,085|[Google Drive](https://drive.google.com/file/d/1WdQ62XJuvw0_K4MUP5nXOhv2RsEBVB1f/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1iN68cdiPO0bTTN_hwmbe9w)|
|[IJB-B](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Whitelam_IARPA_Janus_Benchmark-B_CVPR_2017_paper.pdf)|Raw|1,845|21,798|55,026|7,011|[Google Drive](https://drive.google.com/file/d/15oibCHL3NX-q-QV8q_UAmbIr9e_M0n1R/view?usp=sharing)|
|[CFP](http://www.cfpw.io/paper.pdf)|Raw|500|7,000|-|-|[Google Drive](https://drive.google.com/file/d/1tGNtqzWeUx3BYAxRHBbH1Wy7AmyFtZkU/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/10Qq64LO_RWKD2cr_D32_6A)|
|[CFP](http://www.cfpw.io/paper.pdf)|Align_112x112|500|7,000|-|-|[Google Drive](https://drive.google.com/file/d/1-sDn79lTegXRNhFuRnIRsgdU88cBfW6V/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1DpudKyw_XN1Y491n1f-DtA)|
|[Umdfaces](https://arxiv.org/pdf/1611.01484.pdf)|Align_112x112|8,277|367,888|-|-|[Google Drive](https://drive.google.com/file/d/13IDdIMqPCd8h1vWOYBkW6T5bjAxwmxm5/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1UzrBMguV5YLh8aawIodKeQ)|
|[CelebA](https://arxiv.org/pdf/1411.7766.pdf)|Raw|10,177|202,599|-|-|[Google Drive](https://drive.google.com/file/d/1FO_p759JtKOf3qOnxOGpmoxCcnKiPdBI/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1DfvDKKEB11MrZcf7hPjJfw)|
|[CACD-VS](http://cmlab.csie.ntu.edu.tw/~sirius42/papers/chen14eccv.pdf)|Raw|2,000|163,446|-|-|[Google Drive](https://drive.google.com/file/d/1syrMyJGeXYxbjbmWKLxo1ASzpj2DRrk3/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/13XI67Zn_D_Kncp_9hlTbQQ)|
|[YTF](http://www.cs.tau.ac.il/~wolf/ytfaces/WolfHassnerMaoz_CVPR11.pdf)|Align_344x344|1,595|-|3,425|621,127|[Google Drive](https://drive.google.com/file/d/1o_5b7rYcSEFvTmwmEh0eCPsU5kFmKN_Y/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1M43AcijgGrurb0dfFVlDKQ)|
|[DeepGlint](http://trillionpairs.deepglint.com)|Align_112x112|180,855|6,753,545|-|-|[Google Drive](https://drive.google.com/file/d/1Lqvh24913uquWxa3YS_APluEmbNKQ4Us/view?usp=sharing)|
|[UTKFace](https://susanqq.github.io/UTKFace/)|Align_200x200|-|23,708|-|-|[Google Drive](https://drive.google.com/file/d/1T5KH-DWXu048im0xBuRK0WEi820T28B-/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/12Qp5pdZvitqBYSJHm4ouOw)|
* Remark: unzip [CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf) clean version with 
  ```
  unzip casia-maxpy-clean.zip    
  cd casia-maxpy-clean    
  zip -F CASIA-maxpy-clean.zip --out CASIA-maxpy-clean_fix.zip    
  unzip CASIA-maxpy-clean_fix.zip
  ```
* Remark: after unzip, get image data \& pair ground truths from [AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf), [CFP](http://www.cfpw.io/paper.pdf), [LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf) and [VGGFace2_FP](https://arxiv.org/pdf/1710.08092.pdf) align_112x112 versions with 
  ```python
  import numpy as np
  import bcolz
  import os

  def get_pair(root, name):
      carray = bcolz.carray(rootdir = os.path.join(root, name), mode='r')
      issame = np.load('{}/{}_list.npy'.format(root, name))
      return carray, issame

  def get_data(data_root):
      agedb_30, agedb_30_issame = get_pair(data_root, 'agedb_30')
      cfp_fp, cfp_fp_issame = get_pair(data_root, 'cfp_fp')
      lfw, lfw_issame = get_pair(data_root, 'lfw')
      vgg2_fp, vgg2_fp_issame = get_pair(data_root, 'vgg2_fp')
      return agedb_30, cfp_fp, lfw, vgg2_fp, agedb_30_issame, cfp_fp_issame, lfw_issame, vgg2_fp_issame

  agedb_30, cfp_fp, lfw, vgg2_fp, agedb_30_issame, cfp_fp_issame, lfw_issame, vgg2_fp_issame = get_data(DATA_ROOT)
  ```
* Remark: We share ```MS-Celeb-1M_Top1M_MID2Name.tsv``` ([Google Drive](https://drive.google.com/file/d/15X_mIcmcC38KjHA2NAGUIsNXF_iUeMbX/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1AyZBr_Iow1StS3OzWedT1A)), ```VGGface2_ID2Name.csv``` ([Google Drive](https://drive.google.com/file/d/1tSMrzwkWMCuOycNIjpx9GC3P2Pr1oPOU/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1fRJKvgBxTcd4j6fCfmEUOw)), ```VGGface2_FaceScrub_Overlap.txt``` ([Google Drive](https://drive.google.com/file/d/1M9F29t0WvAIJWhsBn5xyl00VL-7wBYkc/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1ppZ2qcMfZ8bXq5Sf5LHijA)), ```VGGface2_LFW_Overlap.txt``` ([Google Drive](https://drive.google.com/file/d/13MO7su1z0G_Aqc5HwzImBxctORJjyDlO/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1Sl6JIt99oX9G9YwP4HVq2A)), ```CASIA-WebFace_ID2Name.txt``` ([Google Drive](https://drive.google.com/file/d/1Unqo5E5JR2tSNK0g7KhC6uwYM3tTXXVu/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1zlYTOeLRgGPIR-yPeEurRw)), ```CASIA-WebFace_FaceScrub_Overlap.txt``` ([Google Drive](https://drive.google.com/file/d/1xHM6JJXv5cl7xmSbZ1mkXyJpXnEgNV4x/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1smuVPG0j7Zikd7UladBSew)), ```CASIA-WebFace_LFW_Overlap.txt``` ([Google Drive](https://drive.google.com/file/d/1blFEbNGEfncAUQKCeCTb__rv221oZo80/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/10rwsLZFA25e6cW1gJDDZGQ)), ```FaceScrub_Name.txt``` ([Google Drive](https://drive.google.com/file/d/1R8MofI3pXGAuHsD5wswZXLPBHg8zll25/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/12fryVZO6ytpHhjMidvZpxQ)), ```LFW_Name.txt``` ([Google Drive](https://drive.google.com/file/d/1zC-0R3sL_wf2Oq1exMpDvJUGnW0VPcWs/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1OFW8vJajkvTviUMiSNwdXA)), ```LFW_Log.txt``` ([Google Drive](https://drive.google.com/file/d/1afCfVNnguaCaKktsZn8q5CNlqThfeZYk/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1TsQOez_11WcViTV9eo4qOQ)) to help researchers/engineers quickly remove the overlapping parts between their own private datasets and the public datasets.
* Due to release license issue, for other face related databases, please make contact with us in person for more details.


