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
        contents = line[1].split('；')
        info_dict[line[0]] = {
            'high': contents[0]+'。',
            'low': contents[1],
        }
    return info_dict

def get_global_factor_info():
    global_factor_level = {
        '适应与焦虑性': {
            '1': [0, 2],
            '2': [2, 4],
            '3': [4, 6],
            '4': [6, 8],
            '5': [8, 10],
        },
        '内外向性': {
            '1': [0, 2],
            '2': [2, 4],
            '3': [4, 6],
            '4': [6, 8],
            '5': [8, 10],

        },
        '感情用事与安详机警性': {
            '1': [0, 2],
            '2': [2, 4.5],
            '3': [4.5, 5.5],
            '4': [5.5, 8],
            '5': [8, 10],
        },
        '怯懦与果敢性': {
            '1': [0, 2],
            '2': [2, 4],
            '3': [4, 6],
            '4': [6, 8],
            '5': [8, 10],
        },
    }
    global_factor_contents = {
        '适应与焦虑性': {
            '1': '对生活境遇适应性很强，但遇事容易知难而退，缺少艰苦奋斗的动力和毅力。',
            '2': '对当前的生活比较适应，常感到心满意足，能做到所期望的及自认为重要的事情。',
            '3': '对当前的生活比较适应，但仍对人生有更高的期待，能够较好的处理生活境遇与个人希望之间的心理落差，排解内心的焦虑。',
            '4': '对当前的境遇有些不满意，遇到难事易激动，偶尔会感到焦虑。',
            '5': '对生活所要求的和自己意欲达到的事情常感到不满意，容易焦虑，可能会影响工作状态和身体健康。',
        },
        '内外向性': {
            '1': '非常内向，通常羞怯而审慎，沉默寡言，不愿与人打交道。',
            '2': '偏内向，与人相处比较拘谨，常采取克制态度。',
            '3': '中性人格，独处时感到自在，也喜欢与朋友聚餐聊天，需要团队合作时也可以很快与大家打成一片。',
            '4': '偏外向，性格开朗，热情，活泼，适应环境能力强。',
            '5': '非常外向，通常善于交际，与人相处感到能量充沛，不拘小节。',

        },
        '感情用事与安详机警性': {
            '1': '感情丰富，容易产生情绪波动而感到困扰不安，遇到问题常摇摆不定，难做决定。',
            '2': '感情较丰富，对生活中的细节较为含蓄敏感，性格温和，讲究生活艺术，遇事采取行动前会再三思考，顾虑太多。',
            '3': '情绪较稳定，对现实情况有比较清醒的认识，可以较好的控制自己的感情，偶尔也会因为压抑而冲动地发泄情绪。',
            '4': '富有专业心，安详警觉，果断刚毅，但常常过分现实，忽视了许多生活的情趣。',
            '5': '富有进取精神，精力充沛，行动迅速，但有时会考虑不周，不计后果，贸然行事。',
        },
        '怯懦与果敢性': {
            '1': '常人云亦云，个性被动，优柔寡断，受人驱使而不能独立。',
            '2': '性格较懦弱，对他人有一定依赖性，会为获取别人的欢心，而常常迁就他人。',
            '3': '对事物有自己的看法，在自己的专业领域有所为，面对强势也常选择明哲保身，隐藏自己的锋芒。',
            '4': '比较果断独立，常常自动寻找可以施展所长的环境或机会，以充分表现自己的独创能力，并从中获益。',
            '5': '性格果敢，有气魄，锋芒毕露，可能有攻击性倾向。',
        },
    }

    return global_factor_level, global_factor_contents

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

# load personality info
pf16info_file = os.path.join(
    ROOT_DIR,
    'proj',
    'facetraits',
    'personality16info.csv',
)
info16 = read_personality_info(pf16info_file)
global_factor_level, global_factor_contents = get_global_factor_info()

DISPLAY_FACTORS = ['A', 'E', 'F', 'H', 'I', 'L', 'N', 'O', 'Q2', 'Q3']

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
            # plot radar
            sel_scores = {}
            for k in r['scores']:
                if k in DISPLAY_FACTORS:
                    sel_scores[k] = r['scores'][k]
            radar_json = radarplot(sel_scores)
            # filter factor description
            display_factor_info = {}
            for k in DISPLAY_FACTORS:
                _name = FACTORS[k].replace('*', '')
                display_factor_info[_name] = info16[_name]

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
                factors_2nd[fname] = {
                    'value': _v,
                }
                level_ref = global_factor_level[fname]
                contents_ref = global_factor_contents[fname]
                for _l in level_ref:
                    if _v >= level_ref[_l][0] and _v < level_ref[_l][1]:
                        factors_2nd[fname]['level'] = _l
                        factors_2nd[fname]['contents'] = contents_ref[_l]
                        break

            print(time.time() - start_time)
            return render_template(
                'uploaded.html',
                factors_2nd = factors_2nd,
                plotly_data=radar_json,
                info_dict=display_factor_info,
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


