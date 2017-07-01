from flask import Blueprint, request, jsonify, render_template
from ..load import init_model, predictor, get_most_count, most_common
import numpy as np

liner_mod = Blueprint('liner', __name__, template_folder='templates', static_folder='static')

p_model, l_model, graph = init_model()


@liner_mod.route('/liner', methods=['GET','POST'])
def liner():
    if request.method == 'GET':
        return render_template('projects/line.html')
    else:
        query = request.get_data().decode('utf-8')

        text = query.split('\n')
        predict_list = []
        for t in text:
            p = predictor(t)
            predict_list.append(p)

        data = {'AdaBoost': 0,
                'BernoulliNB': 0,
                'DecisionTree': 0,
                'GradientBoost': 0,
                'KNNeighbors': 0,
                'RandomForest': 0,
                'MultinomialNB': 0,
                'MaxEnt': '',
                'SVM': 0,
                '3-layer Perceptron': 0,
                'lstm network': 0}

        # overal per sentence
        predict_list = np.array(predict_list)
        i = 0
        for key in data:
            data[key] = str(get_most_count(predict_list[:, i]))
            i += 1

        # all the sentences with 3 emotions
        predict_list = predict_list.tolist()
        emotion_sents = [0, 0, 0]
        for p in predict_list:
            if most_common(p) == 0:
                emotion_sents[0] += 1
            elif most_common(p) == 1:
                emotion_sents[1] += 1
            else:
                emotion_sents[2] += 1

        # overal score
        score = most_common(list(data.values()))

        return jsonify([data, emotion_sents, score])
