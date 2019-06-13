# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import shutil
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='remove low-shot classes')
    parser.add_argument("-root", "--root",
                        help="specify your dir",
                        default='./data/train',
                        type = str)
    parser.add_argument("-min_num", "--min_num",
                        help="remove the class with less than min_num samples",
                        default=10,
                        type=int)
    args = parser.parse_args()

    # specify your dir
    root = args.root
    # remove the classes with less than min_num samples
    min_num = args.min_num

    # delete '.DS_Store' existed in the source_root
    cwd = os.getcwd()
    os.chdir(root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    for subfolder in os.listdir(root):
        file_num = len(os.listdir(os.path.join(root, subfolder)))
        if file_num <= min_num:
            print("Class {} has less than {} samples, removed!".format(
                                                        subfolder, min_num))
            shutil.rmtree(os.path.join(root, subfolder))
