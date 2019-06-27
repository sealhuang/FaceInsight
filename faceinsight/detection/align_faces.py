# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import argparse
from multiprocessing import Pool
from functools import partial

import numpy as np
from PIL import Image

from faceinsight.detection import detect_faces
from faceinsight.detection.align_trans import get_reference_facial_points
from faceinsight.detection.align_trans import warp_and_crop_face
from faceinsight.detection.matlab_cp2tform import get_similarity_transform
from faceinsight.io.dataset import get_dataset

def worker(image_path, output_class_dir, min_face_size, mode):
    """Sugar function for multiprocessing."""
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_class_dir, filename+'.png')
    #print(image_path)
    if not os.path.exists(output_filename):
        try:
            img = Image.open(image_path).convert('RGB')
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
        else:
            _,landmarks = detect_faces(img, min_face_size=min_face_size,
                                       device=mode)

            # If the landmarks cannot be detected, the img will be discarded
            if len(landmarks)==0:
                print("{} is discarded due to non-detected landmarks!".format(image_path))
                #continue
                return
            # If multiple faces are found, we choose one face which is located
            # center and has larger size
            elif len(landmarks)>1:
                face_weights = []
                for line in landmarks:
                    tmp = [[line[j], line[j + 5]] for j in range(5)]
                    tmp = np.array(tmp)
                    face_center = (tmp.max(axis=0) + tmp.min(axis=0))/2
                    center_diff = face_center - np.array(img.size)/2
                    center_err = np.abs(center_diff[0] * center_diff[1])
                    face_len = tmp.max(axis=0) - tmp.min(axis=0)
                    face_weights.append(face_len[0]*face_len[1]-center_err*2)

                idx = np.argmax(face_weights)
                landmarks = [landmarks[idx]]

            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] 
                                    for j in range(5)]
            ref5points = get_reference_facial_points()
            tfm = get_similarity_transform(ref5points, np.array(facial5points))
            scale = tfm[0][0][0]
            img_width, img_height = img.size
            size_diff = np.array([img_width, img_height]) - \
                        np.array([96*scale, 112*scale])
            nref5points = ref5points * scale + size_diff/2

            warped_face = warp_and_crop_face(np.array(img),
                                             facial5points,
                                             nref5points,
                                             crop_size=(img_width, img_height))
            img_warped = Image.fromarray(warped_face)
            img_warped.save(output_filename)

def main(args):
    """Main function."""
    # output dir config
    output_dir = os.path.expanduser(args.dest_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get uncropped face images
    ds = get_dataset(args.source_dir, has_class_directories=args.has_class_dirs)
    total_img_num = 0
    for cls in ds:
        total_img_num += len(cls)
    
    print('Align faces ...')

    # multiprocessing config
    p = Pool(processes=args.nprocessors)
    for cls in ds:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        p.map(partial(worker, output_class_dir=output_class_dir,
                      min_face_size=args.min_face_size, mode=args.mode),
              cls.image_paths)
    aligned_ds = get_dataset(output_dir,
                             has_class_directories=args.has_class_dirs)
    total_aligned_img_num = 0
    for cls in aligned_ds:
        total_aligned_img_num += len(cls)
    print('Total number of images: %d' % total_img_num)
    print('Number of aligned images: %d' % total_aligned_img_num)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face alignment')
    parser.add_argument('source_dir', type=str,
                        help='specify your source dir')
    parser.add_argument('dest_dir', type=str,
                        help='specify your destination dir')
    parser.add_argument('--min_face_size', type=int,
                        default=35,
                        help='Minimum face size in pixels (35 by default)')
    parser.add_argument('--nprocessors', type=int,
                        default=2,
                        help='Number of workers for multiprocessing.')
    parser.add_argument('--mode', default='cpu', type=str,
                        help='gpu or cpu mode, cpu is the default option')
    parser.add_argument('--has_class_dirs', default=False,
                        help='Has subdirectory for each class',
                        action='store_true')
    args = parser.parse_args()
 
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

