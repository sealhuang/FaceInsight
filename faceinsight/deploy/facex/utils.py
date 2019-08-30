# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import numpy as np
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
from faceinsight.models.shufflefacenet import ShuffleNetV2
from faceinsight.detection import detect_faces
from faceinsight.detection.align_trans import get_reference_facial_points
from faceinsight.detection.align_trans import warp_and_crop_face


class ShuffleFaceNetClfier(nn.Module):
    def __init__(self, class_num):
        super(ShuffleFaceNetClfier, self).__init__()
        self.fc1 = nn.Linear(512, class_num, bias=False)

    def forward(self, x):
        
        x = self.fc1(x)
        return x

class ShuffleFaceNet(nn.Module):
    def __init__(self, backbone_file, clfier_file, device):
        super(ShuffleFaceNet, self).__init__()
        self.backbone = ShuffleNetV2(n_class=512, input_size=224,
                                     width_mult=1.0)
        self.classifier = ShuffleFaceNetClfier(2)
        # load model weights
        self.backbone.load_state_dict(torch.load(backbone_file,
                                     map_location=lambda storage, loc: storage))
        self.classifier.load_state_dict(torch.load(clfier_file,
                                     map_location=lambda storage, loc: storage))
        if device=='gpu':
            self.backbone = self.backbone.cuda()
            self.classifier = self.classifier.cuda()
        self.backbone.eval()
        self.classifier.eval()

    def forward(self, x):
        feat = self.backbone(x)
        x = F.softmax(self.classifier(feat), dim=1)
        return x
 

def load_img(img_file):
    """Load face image."""
    # define transforms
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    test_transform = transforms.Compose([transforms.Resize(250),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    img = Image.open(img_file).convert('RGB')
    img = test_transform(img)
    return img.unsqueeze(0)

def get_square_crop_box(crop_box, box_scalar=1.0):
    """Get square crop box based on bounding box and the expanding scalar.
    Return square_crop_box and the square length.
    """
    center_w = int((crop_box[0]+crop_box[2])/2)
    center_h = int((crop_box[1]+crop_box[3])/2)
    w = crop_box[2] - crop_box[0]
    h = crop_box[3] - crop_box[1]
    box_len = max(w, h)
    delta = int(box_len*box_scalar/2)
    square_crop_box = (center_w-delta, center_h-delta,
                       center_w+delta+1, center_h+delta+1)
    return square_crop_box, 2*delta+1

def crop_face(input_img, output_dir, minsize, scalar, image_size,
              detect_multiple_faces=False, device='cpu'):
    image_path = os.path.expanduser(input_img)
    # output dir config
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    print('Crop face from image ...')

    # returned bounding box
    #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes.txt')
 
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_dir, 'cropped_'+filename+'.png')
    #print(image_path)
    # load image and process
    try:
        img = Image.open(image_path).convert('RGB')
    except (IOError, ValueError, IndexError) as e:
        print('{}: {}'.format(image_path, e))
        return None
    else:
        bounding_boxes, _ = detect_faces(img, min_face_size=minsize,
                                         device=device)
        nrof_faces = len(bounding_boxes)
        if nrof_faces>0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.size)
            #img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                # if multiple faces found, we choose one face
                # which is located center and has larger size
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[0], (det[:,1]+det[:,3])/2-img_center[1] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    # some extra weight on the centering
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb, box_size = get_square_crop_box(det, scalar)
                # get the valid pixel index of cropped face
                face_left = np.maximum(bb[0], 0)
                face_top = np.maximum(bb[1], 0)
                face_right = np.minimum(bb[2], img_size[0])
                face_bottom = np.minimum(bb[3], img_size[1])
                # cropped square image
                new_img = Image.new('RGB', (box_size, box_size))
                # fullfile the cropped image
                cropped = img.crop([face_left, face_top,
                                    face_right, face_bottom])
                w_start_idx = np.maximum(-1*bb[0], 0)
                h_start_idx = np.maximum(-1*bb[1], 0)
                new_img.paste(cropped,(w_start_idx,h_start_idx))
                scaled = new_img.resize((image_size, image_size),
                                        Image.BILINEAR)
                filename_base,file_extension = os.path.splitext(output_filename)
                if detect_multiple_faces:
                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                else:
                    output_filename_n = "{}{}".format(filename_base, file_extension)
                scaled.save(output_filename_n)
                return output_filename_n
        else:
            print('Unable to crop "%s"' % image_path)
            return None

def align_face(input_img, image_size, scalar):
    image_path = input_img
    
    print('Align faces ...')
    # specify size of aligned faces, align and crop with padding
    # due to the bounding box was expanding by a scalar, the `real` face size
    # should be corrected
    scale = image_size * 1.0 /scalar / 112.
    offset = image_size * (scalar - 1.1) / 2
    reference = get_reference_facial_points(default_square=True)*scale + offset

    output_dir = os.path.split(image_path)[0]
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_dir, 'aligned_'+filename+'.png')
    #print(image_path)
    try:
        img = Image.open(image_path).convert('RGB')
    except (IOError, ValueError, IndexError) as e:
        print('{}: {}'.format(image_path, e))
        return None
    else:
        _, landmarks = detect_faces(img, device='cpu')

        # If the landmarks cannot be detected, the img will be discarded
        if len(landmarks)==0: 
            print("{} is discarded due to non-detected landmarks!".format(image_path))
            return None
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] 
                            for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img),
                                         facial5points,
                                         reference,
                                         crop_size=(image_size, image_size))
        img_warped = Image.fromarray(warped_face)
        img_warped.save(output_filename)
        return output_filename

