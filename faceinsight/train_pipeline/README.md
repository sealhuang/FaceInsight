# Pipeline for Training a Face Recognition Model based on PyTorch

****
### Usage 

* `mkdir data checkpoint log` at appropriate directory to store your train/val/test data, checkpoints and training logs.
* Prepare your train/val/test data (refer to Sec. [Data Zoo](#Data-Zoo) for publicly available face related databases), and ensure each database folder has the following structure:
  ```
  ./data/db_name/
          -> id1/
              -> 1.jpg
              -> ...
          -> id2/
              -> 1.jpg
              -> ...
          -> ...
              -> ...
              -> ...
  ```
* Refer to the codes of corresponding sections for specific purposes.

****
### Face Detection and Alignment 

* This section is based on the work of [MTCNN](https://arxiv.org/pdf/1604.02878.pdf).
* Folder: ```./detection```
* Face detection, landmark localization APIs and visualization toy example with ipython notebook:
  ```python 
  from PIL import Image
  from detector import detect_faces
  from visualization_utils import show_results

  img = Image.open('some_img.jpg') # modify the image path to yours
  bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
  show_results(img, bounding_boxes, landmarks) # visualize the results
  ``` 
* Face alignment API (perform face detection, landmark localization and alignment with affine transformations on a whole database folder ```source_root``` with the directory structure as demonstrated in Sec. [Usage](#Usage), and store the aligned results to a new folder ```dest_root``` with the same directory structure): 
  ```
  python face_align.py -source_root [source_root] -dest_root [dest_root] -crop_size [crop_size]

  # python face_align.py -source_root './data/test' -dest_root './data/test_Aligned' -crop_size 112
  ```
* For macOS users, there is no need to worry about ```*.DS_Store``` files which may ruin your data, since they will be automatically removed when you run the scripts.
* Keynotes for customed use: 1) specify the arguments of ```source_root```, ```dest_root``` and ```crop_size``` to your own values when you run ```face_align.py```; 2) pass your customed ```min_face_size```, ```thresholds``` and ```nms_thresholds``` values to the ```detect_faces``` function of ```detector.py``` to match your practical requirements; 3) if you find the speed using face alignment API is a bit slow, you can call face resize API to firstly resize the image whose smaller size is larger than a threshold (specify the arguments of ```source_root```, ```dest_root``` and ```min_side``` to your own values) before calling the face alignment API:
  ```
  python face_resize.py
  ```

****
### Data Processing 

* Remove low-shot data API (remove the low-shot classes with less than ```min_num``` samples in the training set ```root``` with the directory structure as demonstrated in Sec. [Usage](#Usage) for data balance and effective model training):
  ```
  python remove_lowshot.py -root [root] -min_num [min_num]

  # python remove_lowshot.py -root './data/train' -min_num 10
  ```
* Keynotes for customed use: specify the arguments of ```root``` and ```min_num``` to your own values when you run ```remove_lowshot.py```.
* We prefer to include other data processing tricks, *e.g.*, augmentation (flip horizontally, scale hue/satuation/brightness with coefficients uniformly drawn from \[0.6,1.4\], add PCA noise with a coefficient sampled from a normal distribution N(0,0.1), *etc.*), weighted random sampling, normalization, *etc.* to the main training script in Sec. [Training and Validation](#Training-and-Validation) to be self-contained.


****
### Model Zoo 

* Model

  |Backbone|Head|Loss|Training Data|Download Link|
  |:---:|:---:|:---:|:---:|:---:|
  |[IR-50](https://arxiv.org/pdf/1512.03385.pdf)|[ArcFace](https://arxiv.org/pdf/1801.07698.pdf)|[Focal](https://arxiv.org/pdf/1708.02002.pdf)|[MS-Celeb-1M_Align_112x112](https://arxiv.org/pdf/1607.08221.pdf)|[Google Drive](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1L8yOF1oZf6JHfeY9iN59Mg)|

  * Setting
    ```
    INPUT_SIZE: [112, 112]; RGB_MEAN: [0.5, 0.5, 0.5]; RGB_STD: [0.5, 0.5, 0.5]; BATCH_SIZE: 512 (drop the last batch to ensure consistent batch_norm statistics); Initial LR: 0.1; NUM_EPOCH: 120; WEIGHT_DECAY: 5e-4 (do not apply to batch_norm parameters); MOMENTUM: 0.9; STAGES: [30, 60, 90]; Augmentation: Random Crop + Horizontal Flip; Imbalanced Data Processing: Weighted Random Sampling; Solver: SGD; GPUs: 4 NVIDIA Tesla P40 in Parallel
    ```
  * Training \& validation statistics
  
    <img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig13.png" width="1000px"/>
      
  * Performance

    |[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|[CFP_FF](http://www.cfpw.io/paper.pdf)|[CFP_FP](http://www.cfpw.io/paper.pdf)|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |99.78|99.69|98.14|97.53|95.87|92.45|95.22|

* Model

  |Backbone|Head|Loss|Training Data|Download Link|
  |:---:|:---:|:---:|:---:|:---:|
  |[IR-50](https://arxiv.org/pdf/1512.03385.pdf)|[ArcFace](https://arxiv.org/pdf/1801.07698.pdf)|[Focal](https://arxiv.org/pdf/1708.02002.pdf)|Private Asia Face Data|[Google Drive](https://drive.google.com/drive/folders/11TI4Gs_lO-fbts7cgWNqvVfm9nps2msE?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/18BSUeA1bpAeRWTprtHgX9w)|

  * Setting
    ```
    INPUT_SIZE: [112, 112]; RGB_MEAN: [0.5, 0.5, 0.5]; RGB_STD: [0.5, 0.5, 0.5]; BATCH_SIZE: 1024 (drop the last batch to ensure consistent batch_norm statistics); Initial LR: 0.01 (initialize weights from the above model pre-trained on MS-Celeb-1M_Align_112x112); NUM_EPOCH: 80; WEIGHT_DECAY: 5e-4 (do not apply to batch_norm parameters); MOMENTUM: 0.9; STAGES: [20, 40, 60]; Augmentation: Random Crop + Horizontal Flip; Imbalanced Data Processing: Weighted Random Sampling; Solver: SGD; GPUs: 8 NVIDIA Tesla P40 in Parallel
    ```

  * Performance (please perform evaluation on your own Asia face benchmark dataset)


