# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import argparse
from faceinsight.util.mxloader import load_bin, load_mx_rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extracting faces_emore data')
    parser.add_argument('-r', '--rec_path',
                        help='mxnet record file path',
                        default='./faces_emore',
                        type = str)
    args = parser.parse_args()
    rec_path = os.path.abspath(args.rec_path)
    load_mx_rec(rec_path)
    
    bin_files = ['agedb_30', 'calfw', 'cfp_ff', 'cfp_fp', 'cplfw', 'lfw',
                 'vgg2_fp']
    
    for i in range(len(bin_files)):
        load_bin(os.path.join(rec_path, bin_files[i]+'.bin'),
                 os.path.join(rec_path, bin_files[i]))

