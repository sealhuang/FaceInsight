# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np

import plotly.graph_objects as go
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from flask import request, render_template
from werkzeug.utils import secure_filename

from utils import ShuffleFaceNet
from utils import crop_face
from utils import align_face
from utils import load_img


ALLOW_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSIONS

def _eval(model, face_data):
    out = model(face_data)
    out = out.cpu().data.numpy()[0][1]
    return out

class Predictor():
    def __init__(self, factor, model_dir, device):
        self.factor_name = factor
        self.device = device
        # load models
        self.models = []
        for i in range(5):
            model_file_prefix = 'finetuned_shufflenet4%s_'%(factor.lower())
            file_list = os.listdir(model_dir)
            backbone_file = [os.path.join(model_dir, item) for item in file_list
                    if item.startswith(model_file_prefix+'backbone_f%s'%(i))][0]
            clfier_file = [os.path.join(model_dir, item) for item in file_list
                    if item.startswith(model_file_prefix+'clfier_f%s'%(i))][0]
            self.models.append(ShuffleFaceNet(backbone_file, clfier_file,
                                              device))

    def run(self, face_data):
        outs = []
        if self.device=='gpu':
            face_data = face_data.cuda()
        for model in self.models:
            outs.append(_eval(model, face_data))

        return {self.factor_name: np.median(outs)}

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
                      showlegend=False)
    fig_json = fig.to_json()
    return fig_json


# general config
UPLOAD_FOLDER = os.path.expanduser('~/Downloads/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)
root_dir = '/home/huanglj/repo/FaceInsight/faceinsight'
#root_dir = '/Users/sealhuang/repo/FaceInsight/faceinsight'
model_dir = os.path.join(root_dir, 'proj', 'facetraits',
                         '16pfmodels_shufflefacenet')
pf16info_file = os.path.join(root_dir, 'proj','facetraits',
                             'personality16info.csv') 
DEVICE_DET = 'cpu'
DEVICE_CLS = 'gpu'

# load personality info
info16 = read_personality_info(pf16info_file)

# get predictors
FACTORS = {'A': '乐群性*',
           'B': '聪慧性',
           'C': '稳定性',
           'E': '恃强性',
           'F': '兴奋性*',
           'G': '有恒性',
           'H': '敢为性*',
           'I': '敏感性*',
           'L': '怀疑性*',
           'M': '幻想性',
           'N': '世故性*',
           'O': '忧虑性',
           'Q1': '实验性',
           'Q2': '独立性*',
           'Q3': '自律性*',
           'Q4': '紧张性'}
#FACTORS = {'A': '乐群性'}


# load predictor for each personality factor
predictors = []
for factor in FACTORS:
    predictors.append(Predictor(factor, model_dir, DEVICE_CLS))

# initialize the app
app = Flask(__name__, template_folder='./')
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method=='POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if f.filename=='':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            start_time = time.time()
            filename = secure_filename(f.filename)
            filename = str(int(time.time())) +'.' + filename.split('.')[-1]
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(saved_path)
            #return redirect(url_for('uploaded_file', filename=filename))
            # crop face
            crop_face_file = crop_face(saved_path, app.config['UPLOAD_FOLDER'],
                                       50, 1.4, 224,
                                       detect_multiple_faces=False,
                                       device=DEVICE_DET)
            if crop_face_file:
                aligned_face_file = align_face(crop_face_file, 224, 1.4)
                if aligned_face_file:
                    face_data = load_img(aligned_face_file)
                    # eval
                    scores = {}
                    for predictor in predictors:
                        res = predictor.run(face_data)
                        scores.update(res)
                    print(time.time()-start_time)
                    radar_json = radarplot(scores)
                    return render_template('result.html',
                                           plotly_data=radar_json,
                                           info_dict=info16,
                                           imgpath=url_for('uploaded_file',
                                filename=os.path.basename(aligned_face_file)))
            else:
                flash('No face detected')
                return redirect(request.url)

    return """
    <!doctype html>
    <title>Upload Face Image</title>
    <h1>Upload Face Image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


if __name__=='__main__':
    # init process pool
    #--------- RUN WEB APP SERVER ------------#
    # Start the app server on port 5000
    #app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='192.168.1.10', port=5000, debug=True)


