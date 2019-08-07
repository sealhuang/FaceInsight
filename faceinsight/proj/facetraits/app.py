# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask import request, render_template
from model_test import load_ensemble_vggfacenet
from model_test import load_ensemble_shufflenet
from model_test import crop_face
from model_test import align_face
from model_test import face_eval


UPLOAD_FOLDER = os.path.expanduser('~/Downloads/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)
ALLOW_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEVICE = 'cpu'

# load model
FACTOR = 'A'
#ensemble_model = load_ensemble_model(FACTOR, DEVICE)
ensemble_model = load_ensemble_shufflenet(FACTOR, DEVICE)

# initialize the app
app = Flask(__name__, template_folder='./')
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSIONS

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
            filename = secure_filename(f.filename)
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(saved_path)
            #return redirect(url_for('uploaded_file', filename=filename))
            # eval
            crop_face_file = crop_face(saved_path, app.config['UPLOAD_FOLDER'],
                                       20, 1.4, 224,
                                       detect_multiple_faces=False,
                                       device=DEVICE)
            if crop_face_file:
                aligned_face_file = align_face(crop_face_file, 224, 1.4)
                if aligned_face_file:
                    score = face_eval(aligned_face_file, ensemble_model, DEVICE)
                    return render_template('result.html', pred=score,
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


#--------- RUN WEB APP SERVER ------------#
# Start the app server on port 80
# (The default website port)
app.run(host='127.0.0.1', port=5000, debug=True)
#app.run(host='192.168.0.152', port=5000, debug=True)
