# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import time
import streamlit as st
import cv2 as cv
from PIL import Image
import torch

from faceinsight.detection import MTCNNDetector, show_bboxes
from faceinsight.detection.align_trans import get_reference_facial_points
from faceinsight.detection.align_trans import warp_and_crop_face



#-- gender and age classifier
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)',
            '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def initialize_caffe_models():
    age_net = cv.dnn.readNetFromCaffe('data/deploy_age.prototxt', 
                                      'data/age_net.caffemodel')

    gender_net = cv.dnn.readNetFromCaffe('data/deploy_gender.prototxt',
                                         'data/gender_net.caffemodel')

    return age_net, gender_net

#-- face crop apis
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

def crop_face(img, minsize, scalar, image_size, face_detector,
              detect_multiple_faces=True):
    """Crop and align faces.

    Return:
        face bounding boxes: list of coordinates
        cropped faces: list of images
    """
    #print('Crop face from image ...')
    bounding_boxes, _ = face_detector.infer(img, min_face_size=minsize)
    nrof_faces = len(bounding_boxes)
    faces = []
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
            _, landmarks = face_detector.infer(new_img, min_face_size=image_size/2)
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
            #img_warped = Image.fromarray(warped_face)
            #faces.append(img_warped)
            faces.append(warped_face)
    
    return bounding_boxes, faces

#-- main
st.title('Face Recognition Demo')

# init face detector
detector = MTCNNDetector(device='cpu')

# init classifier
age_net, gender_net = initialize_caffe_models()

@st.cache(allow_output_mutation=True)
def get_cap():
    return cv.VideoCapture(0)

cap = get_cap()

frameST = st.empty()
faceST = st.empty()
#param=st.sidebar.slider('chose your value')

last_time = time.time()
while True:
    ret, frame = cap.read()

    # Stop the program if press `q`
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Done processing !!!")
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # crop center square from 1280x960 image
    frame = frame[:, 160:-160, :]
    # BGR to RGB, and switch the left and right side
    im = Image.fromarray(frame[:, ::-1, ::-1])
    im = im.resize((int(im.width/2), int(im.height/2)))
    # face detction
    bounding_boxes, _ = detector.infer(im, min_face_size=80)
    face_im = show_bboxes(im, bounding_boxes)
    face_im = np.array(face_im)
    frameST.image(face_im, channels="RGB")

    # gender and age
    font = cv.FONT_HERSHEY_SIMPLEX
    if (time.time()-last_time)>3:
        last_time = time.time()
        bounding_boxes, faces = crop_face(im, minsize=80, scalar=1.2,
                                          image_size=227,
                                          face_detector=detector)
        #print("Found {} faces".format(str(len(faces))))
        for i in range(len(faces)):
            blob = cv.dnn.blobFromImage(faces[i][:, :, ::-1], 1, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : %s, p : %s"%(gender, max(gender_preds[0])))

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)

            overlay_text = "%s %s" % (gender, age)
            cv.putText(face_im, overlay_text,
                       (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])),
                       font, 1, (255, 255, 255), 2, cv.LINE_AA)
        faceST.image(face_im, channels="RGB")


#import time
#
#'Starting a long computation...'
#
#latest_iteration = st.empty()
#bar = st.progress(0)
#
#for i in range(100):
#    latest_iteration.text(f'Iteration {i+1}')
#    bar.progress(i+1)
#    time.sleep(0.1)
#
#'... and now we\'re done!'
#
#st.write('Here is our first attemp at using data to create a table:')
#st.write(pd.DataFrame({
#    'first column': [1, 2, 3, 4],
#    'second column': [10, 20, 30, 40]
#    }))
#
#df = pd.DataFrame({
#    'first column': [1, 2, 3, 4],
#    'second column': [10, 20, 30, 40]
#    })
#
#chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])
#st.line_chart(chart_data)
#
#map_data = pd.DataFrame(
#        np.random.randn(1000, 2)/[50, 50] + [37.76, -122.4],
#        columns=['lat', 'lon'])
#st.map(map_data)
#
#if st.checkbox('Show dataframe'):
#    chart_data = pd.DataFrame(
#            np.random.randn(20, 3),
#            columns=['a', 'b', 'c'])
#    st.line_chart(chart_data)
#
##option = st.selectbox(
##        'Which number do you like best?',
##        df['first column'])
##
##'You selected: ', option
#
#option = st.sidebar.selectbox(
#        'Which number do you like best?',
#        df['first column'])
#
#'You selected: ', option

