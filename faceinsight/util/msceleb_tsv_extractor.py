# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Utils for Extracting images from MS-Celeb-1M tsv file."""

import os
import base64
import struct

root_dir = '/home/huanglj/database/MS-Celeb-1M'
fid = open(os.path.join(root_dir, 'MsCelebV1_Faces_Aligned.tsv'), 'r')
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
        filename = data_info[0] +'/'+ data_info[1] +'-'+ data_info[4] + '.jpg'
        bbox = struct.unpack('ffff', data_info[5].decode("base64"))
        bbox_file.write(filename + ' ' + 
                    (' '.join(str(bbox_value) for bbox_value in bbox)) + '\n')

        img_data = data_info[6].decode("base64")
        output_file_path = os.path.join(img_dir, filename)
        if os.path.exists(output_file_path):
            print output_file_path + " exists"

        output_path = os.path.dirname(output_file_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        img_file = open(output_file_path, 'w')
        img_file.write(img_data)
        img_file.close()
    else:
        break

bbox_file.close()
fid.close()

