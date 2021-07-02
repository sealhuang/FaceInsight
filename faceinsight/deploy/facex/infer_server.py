# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import io
import time

import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

from config import *
from utils import ShuffleFaceNet
from utils import img_preprocess


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
            out = model(face_data)
            outs.append(out.cpu().data.numpy()[0][1])

        return {self.factor_name: float(np.median(outs))}

# initialize the app
app = Flask(__name__)
#app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# general config
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)

model_dir = os.path.join(ROOT_DIR, 'proj', 'facetraits',
                         '16pfmodels_shufflefacenet')
DEVICE = 'gpu'

# load predictor for each personality factor
FACTOR_LIST = [
    'A', 'B', 'C', 'E',
    'F', 'G', 'H', 'I',
    'L', 'M', 'N', 'O',
    'Q1', 'Q2', 'Q3', 'Q4',
]
predictors = []
for factor in FACTOR_LIST:
    predictors.append(Predictor(factor, model_dir, DEVICE))

@app.route('/predict', methods=['POST'])
def predict():
    """Main function for facial personality inference."""
    # initialize the data dictionary that will be returned
    data = {'success': False}

    # ensure an image was properly uploaded to our endpoint
    if request.method=='POST':
        if request.files.get('image'):
            # read the image in PIL format
            image = request.files['image'].read()
            try:
                img = Image.open(io.BytesIO(image)).convert('RGB')
                img = img_preprocess(img)
            except (IOError, ValueError, IndexError) as e:
                print('Image loading error: {}'.format(e))
            else:
                data['scores'] = {}
                for predictor in predictors:
                    res = predictor.run(img)
                    data['scores'].update(res)
                # indicate that the request was a success
                data['success'] = True
   
    # return the data dictionary as a JSON response
    return jsonify(data)


if __name__=='__main__':
    #--------- RUN WEB APP SERVER ------------#
    # Start the app server on port 5003
    app.run(host=INF_URL, port=5003, debug=True)


