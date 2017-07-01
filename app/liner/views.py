from flask import Blueprint, request, jsonify, render_template
from ..load import init_model, predictor

liner_mod = Blueprint('liner', __name__, template_folder='templates', static_folder='static')

p_model, l_model, graph = init_model()


@liner_mod.route('/liner', methods=['GET','POST'])
def liner():
    if request.method == 'GET':
        return render_template('projects/line.html')
    else:
        query = request.get_data().decode('utf-8')
        result = predictor(query)
        return jsonify(result)
