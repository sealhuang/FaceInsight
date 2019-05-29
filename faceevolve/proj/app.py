# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask import request, render_template
#import catdog

UPLOAD_FOLDER = os.path.expanduser('~/Downloads/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)
ALLOW_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# initialize the app
app = Flask(__name__)
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
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """

#@app.route("/uploader", methods=["GET", "POST"])
#def get_image():
#    if request.method == 'POST':
#        f = request.files['file']
#        sfname = 'static/images/'+str(secure_filename(f.filename))
#        f.save(sfname)
#
#        clf = catdog.classifier()
#        #clf.save_image(f.filename)
#
#        return render_template('result.html',
#                               pred=clf.predict(sfname),
#                               imgpath=sfname)


#--------- RUN WEB APP SERVER ------------#
# Start the app server on port 80
# (The default website port)
app.run(host='127.0.0.1', port=5000, debug=True)