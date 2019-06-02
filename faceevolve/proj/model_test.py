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
from faceevolve.backbone import model_vgg_face
from faceevolve.align import detect_faces
from faceevolve.align.align_trans import get_reference_facial_points
from faceevolve.align.align_trans import warp_and_crop_face

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


def load_img(img_file):
    # define transforms
    normalize = transforms.Normalize(mean=[0.518, 0.493, 0.506],
                                     std=[0.270, 0.254, 0.277])
    test_transform = transforms.Compose([transforms.Resize(250),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    img = Image.open(img_file).convert('RGB')
    img = test_transform(img)
    return img.unsqueeze(0)

def load_model(model_file, device):
    model_backbone = model_vgg_face.VGG_Face_torch
    model = clsNet1(model_backbone, 2)
    model.load_state_dict(torch.load(model_file, map_location=device))
    return model

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
              detect_multiple_faces=False):
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
        bounding_boxes, _ = detect_faces(img, min_face_size=minsize)
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
        _, landmarks = detect_faces(img)

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

def load_ensemble_model(factor):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load model
    ensemble_models = []
    for i in range(5):
        model_file_prefix = 'finetuned_vggface4%s'%(factor.lower())
        file_list = os.listdir('./model_weights')
        model_file = [os.path.join('./model_weights',item) for item in file_list
                        if item.startswith(model_file_prefix+'_f%s'%(i))][0]
        model = load_model(model_file, device)
        ensemble_models.append(model.to(device))

    return ensemble_models

def face_eval(face_file, ensemble_model):
    # load image
    img_data = load_img(face_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output = []
    # face eval
    for model in ensemble_model:
        model.eval()
        data = img_data.to(device)
        out = F.softmax(model(data), dim=1)
        out = out.cpu().data.numpy()[0][1]
        output.append(out)
    return np.mean(output)

def main(args):
    input_img = args.input_img
    output_dir = args.output_dir
    min_face_size = args.min_face_size
    detect_multiple_faces = args.detect_multiple_faces
    scalar = args.scalar
    image_size = args.image_size

    # load model
    factor = 'A'
    ensemble_model = load_ensemble_model(factor)

    # crop face from input image
    crop_face_file = crop_face(input_img, output_dir, min_face_size,
                               scalar, image_size,
                               detect_multiple_faces=detect_multiple_faces)

    if crop_face_file:
        aligned_face_file = align_face(crop_face_file, image_size, scalar)
        if aligned_face_file:
            score = face_eval(aligned_face_file, ensemble_model)
            print('Factor %s, score: %s'%(factor, score))
            return score
    else:
        return None

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
 
    parser.add_argument('input_img',
                        type=str,
                        help='uncropped face images.')
    parser.add_argument('output_dir',
                        type=str,
                        help='Directory with cropped face thumbnails.')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Image size in pixels (224 by default).')
    parser.add_argument('--min_face_size',
                        type=int,
                        default=20,
                        help='Minimum face size in pixels (35 by default).')
    parser.add_argument('--scalar',
                        type=float,
                        default=1.4,
                        help='expanding scaler for the bounding box (1.4 by default).')
    parser.add_argument('--detect_multiple_faces',
                        default=False,
                        help='Detect and align multiple faces per image.',
                        action='store_true')
    return parser.parse_args(argv)

def batch_test():
    root_dir = r'/Users/sealhuang/project/faceTraits/bnuData'
    img_dir = os.path.join(root_dir, 'anony_pics')
    test_file = os.path.join(root_dir, 'mbti_workbench', 'unique_mbti_ei.csv')
    test_list = open(test_file).readlines()
    test_list.pop(0)
    test_list = [line.strip().split(',') for line in test_list]
    test_score = []
    for line in test_list[:1000]:
        msked_id = int(line[0])
        img_file = os.path.join(img_dir, format(msked_id, '012d')+'.jpg')
        print(img_file)
        argv = [img_file, '~/Downloads/test']
        s = main(parse_arguments(argv))
        if not s:
            s = 'NaN'
        test_score.append(s)

        with open('mbti_x2_test_score.csv', 'a+') as f:
            f.write(','.join([line[0], line[1], str(s)])+'\n')


if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    batch_test()

