# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import argparse
import random
from time import sleep

import numpy as np
from PIL import Image
from scipy import misc

from faceinsight.detection import detect_faces
from faceinsight.io.dataset import get_dataset


def get_square_crop_box(crop_box, box_scaler=1.0):
    """Get square crop box based on bounding box and the expanding scaler.
    Return square_crop_box and the square length.
    """
    center_w = int((crop_box[0]+crop_box[2])/2)
    center_h = int((crop_box[1]+crop_box[3])/2)
    w = crop_box[2] - crop_box[0]
    h = crop_box[3] - crop_box[1]
    box_len = max(w, h)
    delta = int(box_len*box_scaler/2)
    square_crop_box = (center_w-delta, center_h-delta,
                       center_w+delta+1, center_h+delta+1)
    return square_crop_box, 2*delta+1

def main(args):
    sleep(random.random())
    
    # output dir config
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get uncropped face images
    ds = get_dataset(args.input_dir, has_class_directories=args.has_class_dirs)
    
    print('Crop faces from image ...')
    
    # minimum size of face
    minsize = args.min_face_size

    # returned bounding box
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes.txt')
 
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cls in ds:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir,filename+'.png')
                #print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = Image.open(image_path).convert('RGB')
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        bounding_boxes, _ = detect_faces(img,
                                                         min_face_size=minsize,
                                                         device=args.mode)
                        nrof_faces = len(bounding_boxes)
                        if nrof_faces>0:
                            det = bounding_boxes[:, 0:4]
                            det_arr = []
                            img_size = np.asarray(img.size)
                            #img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces>1:
                                if args.detect_multiple_faces:
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
                                bb, box_size = get_square_crop_box(det,
                                                                   args.scaler)
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
                                scaled = new_img.resize((args.image_size,
                                                         args.image_size),
                                                        Image.BILINEAR)
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                scaled.save(output_filename_n)
                                #misc.imsave(output_filename_n, scaled)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to crop "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d'%nrof_successfully_aligned)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir',
                        type=str,
                        help='Directory with uncropped face images.')
    parser.add_argument('output_dir',
                        type=str,
                        help='Directory with cropped face thumbnails.')
    parser.add_argument('--has_class_dirs',
                        default=False,
                        help='Has subdirectory for each class.',
                        action='store_true')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Image size in pixels (224 by default).')
    parser.add_argument('--min_face_size',
                        type=int,
                        default=35,
                        help='Minimum face size in pixels (35 by default).')
    parser.add_argument('--scaler',
                        type=float,
                        default=1.4,
                        help='expanding scaler for the bounding box (1.4 by default).')
    parser.add_argument('--mode', default='cpu', type=str,
                        help='cpu or gpu mode, cpu is the default option')
    parser.add_argument('--detect_multiple_faces',
                        default=False,
                        help='Detect and align multiple faces per image.',
                        action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

