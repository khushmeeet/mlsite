from flask import Blueprint, request, jsonify, render_template
from ..load import processing_results

liner_mod = Blueprint('liner', __name__, template_folder='templates', static_folder='static')


@liner_mod.route('/liner', methods=['GET','POST'])
def liner():
    if request.method == 'GET':
        return render_template('projects/line.html')
    else:
        query = request.get_data().decode('utf-8')
        data, emotion_sents, score = processing_results(query)

        return jsonify([data, emotion_sents, score])
