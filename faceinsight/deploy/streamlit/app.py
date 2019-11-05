# vi: set ft=python sts=4 ts=4 sw=4 et:

import streamlit as st
import numpy as np
import cv2 as cv

st.title('Face Recognition Demo')


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv.VideoCapture(0)

cap = get_cap()

frameST = st.empty()
param=st.sidebar.slider('chose your value')

while True:
    ret, frame = cap.read()
    # Stop the program if reached end of video
    if not ret:
        print("Done processing !!!")
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    frameST.image(frame, channels="BGR")

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

