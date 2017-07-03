from flask import Blueprint, request, jsonify, render_template
from ..load import processing_results
from memory_profiler import profile

liner_mod = Blueprint('liner', __name__, template_folder='templates', static_folder='static')

@profile
@liner_mod.route('/liner', methods=['GET', 'POST'])
def liner():
    if request.method == 'POST':
        query = request.form['liner-text']
        data, emotion_sents, score, line_sentiment, text = processing_results(query)
        return render_template('projects/line.html', data=[data, emotion_sents, score, zip(text, line_sentiment)])
    else:
        return render_template('projects/line.html')
