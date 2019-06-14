# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Utils for Extracting images from MS-Celeb-1M tsv file."""

import os
import base64
import struct

def tsv_extractor(root_dir, tsv_file):
    """Extract images from tsv file."""
    fid = open(os.path.join(root_dir, tsv_file), 'r')
    db_dir = os.path.join(root_dir, 'extracted')
    img_dir = os.path.join(db_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, mode=0o755)
    bbox_file = open(db_dir + '/bboxes.txt', 'w')
    while True:
        line = fid.readline()
        if line:
            data_info = line.split('\t')
            # 0: Freebase MID (unique key for each entity)
            # 1: ImageSearchRank
            # 4: FaceID
            # 5: bbox
            # 6: img_data
            filename = data_info[0]+'/'+data_info[1]+'-'+data_info[4]+'.jpg'
            bbox = struct.unpack('ffff', base64.b64decode(data_info[5]))
            bbox_file.write(filename + ' ' + 
                    (' '.join(str(bbox_value) for bbox_value in bbox)) + '\n')

            img_data = base64.b64decode(data_info[6])
            output_file_path = os.path.join(img_dir, filename)
            if os.path.exists(output_file_path):
                print(output_file_path + ' exists')

            output_path = os.path.dirname(output_file_path)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            with open(output_file_path, 'wb') as f:
                f.write(img_data)
        else:
            break

    bbox_file.close()
    fid.close()

def merge_clean_label(root_dir):
    """Merge clean labels based on C-MS-Celeb Cleanlist."""
    label_dir = os.path.join(root_dir, 'C-MS-Celeb-Cleanlist')
    label_file = ['clean_list_128Vec_WT051_P010.txt',
                  'relabel_list_128Vec_T058.txt']
    merged_list = {}
    total_num = 0
    for lf in label_file:
        raw_list = open(os.path.join(label_dir, lf), 'r').readlines()
        raw_list = [line.strip().split() for line in raw_list]
        for line in raw_list:
            if line[0] in merged_list:
                if not line[1] in merged_list[line[0]]:
                    merged_list[line[0]].append(line[1])
                    total_num += 1
                else:
                    continue
            else:
                merged_list[line[0]] = [line[1]]
                total_num += 1

    print('%s unique images found'%(total_num))
    merged_list_file = os.path.join(label_dir, 'merged_clean_list.txt')
    with open(merged_list_file, 'w') as outf:
        class_idx = 0
        for c in merged_list:
            for line in merged_list[c]:
                outf.write('%s %s\n'%(line, class_idx))
            class_idx += 1

def gen_clean_dataset(root_dir, clean_list_file):
    """Copy and organize images based on clean list."""
    img_dir = os.path.join(root_dir, 'extracted', 'imgs')
    clean_dir = os.path.join(root_dir, 'extracted', 'clean')
    
    label_info = open(clean_list_file, 'r').readlines()
    label_info = [line.strip().split() for line in label_info]
    img_idx = 1
    for line in label_info:
        class_dir = os.path.join(clean_dir, line[1])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir, mode=0o755)
        src_img = os.path.join(img_dir, line[0])
        targ_img = os.path.join(class_dir, str(img_idx)+'.jpg')
        os.system(' '.join(['cp', src_img, targ_img]))
        img_idx += 1


if __name__ == '__main__':
    root_dir = '/home/huanglj/database/MS-Celeb-1M'
    tsv_file = 'MsCelebV1_Faces_Aligned.tsv'
    #tsv_extractor(root_dir, tsv_file)
    #merge_clean_label(root_dir)
    # The clean list comes from XiangWu's repo
    # (https://github.com/AlfredXiangWu/face_verification_experiment)
    clean_list_file = os.path.join(root_dir, 'MS-Celeb-1M_clean_list.txt')
    gen_clean_dataset(root_dir, clean_list_file)

