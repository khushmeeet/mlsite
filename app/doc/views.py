from flask import Blueprint, request, render_template, flash, redirect
from ..load import processing_results
from werkzeug.utils import secure_filename
import os

doc_mod = Blueprint('doc', __name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@doc_mod.route('/doc', methods=['GET', 'POST'])
def doc():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.mkdir(UPLOAD_FOLDER)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
        with open('./uploads/' + filename) as f:
            query = f.read()
        data, emotion_sents, score = processing_results(query)

        return render_template('projects/doc.html', data=[data, emotion_sents, score])
    else:
        return render_template('projects/doc.html')