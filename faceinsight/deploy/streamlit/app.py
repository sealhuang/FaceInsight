# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import streamlit as st
import cv2 as cv
from PIL import Image
import torch

from faceinsight.detection.get_nets import PNet, RNet, ONet
from faceinsight.detection.box_utils import nms, calibrate_box
from faceinsight.detection.box_utils import get_image_boxes, convert_to_square
from faceinsight.detection.first_stage import run_first_stage

from faceinsight.detection import show_bboxes
from faceinsight.detection.align_trans import get_reference_facial_points
from faceinsight.detection.align_trans import warp_and_crop_face


# face detection apis
@st.cache(allow_output_mutation=True)
def load_detection_model(device='cpu'):
    """Load face detection model."""
    # load models
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    if device=='gpu':
        pnet = pnet.cuda()
        rnet = rnet.cuda()
        onet = onet.cuda()    
    onet.eval()

    return pnet, rnet, onet

def detect_faces(pnet, rnet, onet,
                 image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7], device='cpu'):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """
    
    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0],
                                device=device)
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    if bounding_boxes:
        bounding_boxes = np.vstack(bounding_boxes)
    else:
        return [], []

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    img_boxes = torch.FloatTensor(img_boxes)
    if device=='gpu':
        img_boxes = img_boxes.cuda()
    with torch.no_grad():
        output = rnet(img_boxes)
    offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0: 
        return [], []
    img_boxes = torch.FloatTensor(img_boxes)
    if device=='gpu':
        img_boxes = img_boxes.cuda()
    with torch.no_grad():
        output = onet(img_boxes)
    landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks


st.title('Face Recognition Demo')

@st.cache(allow_output_mutation=True)
def get_cap():
    return cv.VideoCapture(0)

pnet, rnet, onet = load_detection_model()

cap = get_cap()

frameST = st.empty()
#param=st.sidebar.slider('chose your value')

while True:
    ret, frame = cap.read()

    # Stop the program if reached end of video
    if not ret:
        print("Done processing !!!")
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # BGR to RGB, and switch the left and right side
    im = Image.fromarray(frame[:, ::-1, ::-1])
    im = im.resize((int(im.width/2), int(im.height/2)))
    # face detction
    bounding_boxes, _ = detect_faces(pnet, rnet, onet, im, min_face_size=50, device='cpu')
    face_im = show_bboxes(im, bounding_boxes)

    frameST.image(np.array(face_im), channels="RGB")

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

# vi: set ft=python sts=4 ts=4 sw=4 et:


