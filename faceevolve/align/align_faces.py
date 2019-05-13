# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import numpy as np
import argparse
from PIL import Image

from faceevolve.align import detect_faces
from faceevolve.align.align_trans import get_reference_facial_points
from faceevolve.align.align_trans import warp_and_crop_face
from faceevolve.io.dataset import get_dataset

def main(args):
    # output dir config
    output_dir = os.path.expanduser(args.dest_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # get uncropped face images
    ds = get_dataset(args.source_dir, has_class_directories=False)
    
    print('Align faces ...')
    
    # specify size of aligned faces, align and crop with padding
    # due to the bounding box was expanding by a scalar, the `real` face size
    # should be corrected
    scale = args.image_size * 1.0 /args.expand_scalar / 112.
    offset = args.image_size * (args.expand_scalar - 1.1) / 2
    reference = get_reference_facial_points(default_square=True) * scale + offset

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
                    _, landmarks = detect_faces(img)

                    # If the landmarks cannot be detected, the img will be discarded
                    if len(landmarks)==0: 
                        print("{} is discarded due to non-detected landmarks!".format(image_path))
                        continue
                    facial5points = [[landmarks[0][j], landmarks[0][j + 5]]
                                        for j in range(5)]
                    warped_face = warp_and_crop_face(np.array(img),
                                                     facial5points,
                                                     reference,
                                                     crop_size=(args.image_size,
                                                                args.image_size))
                    img_warped = Image.fromarray(warped_face)
                    img_warped.save(output_filename)
                    nrof_successfully_aligned += 1
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d'%nrof_successfully_aligned)

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face alignment')
    parser.add_argument('source_dir', type=str,
                        help='specify your source dir')
    parser.add_argument('dest_dir', type=str,
                        help='specify your destination dir')
    parser.add_argument('--image_size', type=int, default=224,
                        help='aligned faces size (crop and align with padding)')
    parser.add_argument('--expand_scalar', type=float, default=1.4,
                        help='expanding factor of bounding box')
    args = parser.parse_args()
 
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

