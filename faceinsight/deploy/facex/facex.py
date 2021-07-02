# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import requests

import plotly.graph_objects as go
from flask import Flask, request, redirect, url_for
from flask import flash, render_template
from werkzeug.utils import secure_filename

from config import *


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSIONS

def read_personality_info(info_file):
    """Read personality info of 16PF"""
    info = open(info_file).readlines()
    info = [line.strip().split(':') for line in info]
    info_dict = {}
    for line in info:
        info_dict[line[0]] = line[1]
    return info_dict

def radarplot(data_dict):
    """Plot Radar figure for results."""
    fig = go.Figure()
    # plot personal radar
    ks = [key for key in data_dict.keys()]
    theta = [FACTORS[k] for k in ks]
    r = [data_dict[k] for k in ks]
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself'))
    # plot reference radar
    refr = [0.5]*(len(ks)+1)
    fig.add_trace(go.Scatterpolar(r=refr, theta=theta+[theta[0]],
                                  mode='lines',
                                  line_color='peru',
                                  opacity=0.7,
                                  hoverinfo='none',
                                  name='ref'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                      showlegend=False, width=400, height=400)
    fig_json = fig.to_json()
    return fig_json


# initialize the app
app = Flask(__name__)
#app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# general config
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)

pf16info_file = os.path.join(ROOT_DIR, 'proj','facetraits',
                             'personality16info.csv') 

# load personality info
info16 = read_personality_info(pf16info_file)

def infer_wrapper(url, port, f):
    return requests.post('http://%s:%s/predict'%(url, port), files=f).json()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=='POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        # variables for request parameters
        f = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if f.filename=='':
            print('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            filename = str(int(time.time())) +'.' + filename.split('.')[-1]
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('upload successful')
            return redirect(url_for('inference', filename=filename))
    return render_template('index.html')

@app.route('/inference/<filename>')
def inference(filename):
    start_time = time.time()
    
    # face detection
    image = open(os.path.join(app.config['UPLOAD_FOLDER'],filename),'rb').read()
    payload = {'image': image}
    # submit the request
    r = requests.post('http://%s:5002/predict'%(DET_URL), files=payload).json()
    # ensure the request was successful
    if r['success']:
        face_img = r['faces'][0]
        # facial personality inference
        image = open(os.path.join(app.config['UPLOAD_FOLDER'], face_img),
                     'rb').read()
        payload = {'image': image}
        # submit the request
        r = requests.post('http://%s:5003/predict'%(INF_URL),
                          files=payload).json()
        if r['success']:
            radar_json = radarplot(r['scores'])
            # compute 2nd-order factor score
            wts = {
                '适应与焦虑性': {
                    'OFFSET': 3.8,
                    'L': 2,
                    'O': 3,
                    'Q4': 4,
                    'C': -2,
                    'H': -2,
                    'Q3': -2,
                },
                '内外向性': {
                    'OFFSET': -1.1,
                    'A': 2,
                    'E': 3,
                    'F': 4,
                    'H': 5,
                    'Q2': -2,
                },
                '感情用事与安详机警性': {
                    'OFFSET': 7.7,
                    'C': 2,
                    'E': 2,
                    'F': 2,
                    'N': 2,
                    'A': -4,
                    'I': -6,
                    'M': -2,
                },
                '怯懦与果敢性': {
                    'E': 4,
                    'M': 3,
                    'Q1': 4,
                    'Q2': 4,
                    'A': -3,
                    'G': -2,
                },
            }
            factors_2nd = {}
            for fname in wts:
                _v = 0
                for fidx in wts[fname]:
                    if fidx in r['scores']:
                        _v += r['scores'][fidx]*wts[fname][fidx]
                    else:
                        _v += wts[fname][fidx]
                factors_2nd[fname] = _v

            print(time.time() - start_time)
            return render_template(
                'uploaded.html',
                factors_2nd = factors_2nd,            
                plotly_data=radar_json,
                info_dict=info16,
                filename=face_img,
            )
        else:
            print('Request failed.')
            return render_template('index.html')
    else:
        print('Request failed or no face deteced.')
        return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__=='__main__':
    #--------- RUN WEB APP SERVER ------------#
    # Start the app server on port 5000
    app.run(host=APP_URL, port=5000, debug=True)


