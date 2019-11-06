# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import io
import time

import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

from faceinsight.detection import MTCNNDetector
from faceinsight.detection.align_trans import get_reference_facial_points
from faceinsight.detection.align_trans import warp_and_crop_face

from config import *


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

def crop_face(img, minsize, scalar, image_size,
              detect_multiple_faces=False, device='cpu'):
    """Crop and align faces."""
    #print('Crop face from image ...')
    detector = MTCNNDetector(device=device)
    bounding_boxes, _ = detector.infer(img, min_face_size=minsize)
    nrof_faces = len(bounding_boxes)
    if nrof_faces>0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.size)
        if nrof_faces>1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                # if multiple faces found, we choose one face
                # which is located center and has larger size
                bounding_box_size = (det[:,2]-det[:,0]) * (det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2 - img_center[0],
                                      (det[:,1]+det[:,3])/2 - img_center[1] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                # some extra weight on the centering
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))

        faces = []
        for i, det in enumerate(det_arr):
            #-- crop face first
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
            new_img.paste(cropped, (w_start_idx, h_start_idx))
            new_img = new_img.resize((image_size, image_size), Image.BILINEAR)
            
            #-- face alignment
            # specify size of aligned faces, align and crop with padding
            # due to the bounding box was expanding by a scalar, the `real`
            # face size should be corrected
            scale = image_size * 1.0 / scalar / 112.
            offset = image_size * (scalar - 1.1) / 2
            reference = get_reference_facial_points(default_square=True)*scale \
                        + offset
            _, landmarks = detector.infer(new_img, min_face_size=image_size/2)
            # If the landmarks cannot be detected, the img will be discarded
            if len(landmarks)==0: 
                print('The face is discarded due to non-detected landmarks!')
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] 
                                for j in range(5)]
            warped_face = warp_and_crop_face(np.array(new_img),
                                             facial5points,
                                             reference,
                                             crop_size=(image_size, image_size))
            img_warped = Image.fromarray(warped_face)
            faces.append(img_warped)
        if len(faces):
            return True, faces
        else:
            return False, faces

    else:
        print('No faces detected!')
        return False, []


# initialize the app
app = Flask(__name__)
#app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# general config
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)

root_dir = ROOT_DIR
DEVICE = 'cpu'

@app.route('/predict', methods=['POST'])
def predict():
    """Main function for face detection."""
    # initialize the data dictionary that will be returned
    data = {'success': False}

    # ensure an image was properly uploaded to our endpoint
    if request.method=='POST':
        if request.files.get('image'):
            # read the image in PIL format
            image = request.files['image'].read()
            try:
                img = Image.open(io.BytesIO(image)).convert('RGB')
            except (IOError, ValueError, IndexError) as e:
                print('Image loading error: {}'.format(e))
            else:
                # crop face
                status, face_images = crop_face(img, 50, 1.4, 224,
                                                detect_multiple_faces=False,
                                                device=DEVICE)
                data['faces'] = []
                if status:
                    filename_base = str(int(time.time()*1000))
                    for i in range(len(face_images)):
                        output_filename = "{}_{}.png".format(filename_base, i)
                        face_images[i].save(os.path.join(app.config['UPLOAD_FOLDER'],
                                                         output_filename))
                        data['faces'].append(output_filename)
                    # indicate that the request was a success
                    data['success'] = True
   
    # return the data dictionary as a JSON response
    return jsonify(data)


if __name__=='__main__':
    #--------- RUN WEB APP SERVER ------------#
    # Start the app server on port 5002
    app.run(host=DET_URL, port=5002, debug=True)


