from flask import Flask, jsonify, render_template, request, url_for
from sklearn.externals import joblib
import pickle
import numpy as np
import os
from load import init_model

app = Flask(__name__)

global model, graph

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


def clean(query):
    return vectorizer.transform([query])


def encode(text):
    vector = np.zeros(10577)
    for i, word in enumerate(text.split(' ')):
        try:
            vector[word2index[word]] = 1
        except Exception:
            vector[i] = 0
    return vector


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    query = request.get_data().decode('utf-8')
    clean_query = clean(query)
    ada = adaboost.predict(clean_query)
    ber = bernoulli.predict(clean_query)
    dt = decisiontree.predict(clean_query)
    gb = gradientboost.predict(clean_query.toarray())
    knnp = knn.predict(clean_query)
    rf = randomforest.predict(clean_query)
    mnb = multinomialnb.predict(clean_query)
    svm = svm10.predict(clean_query)

    with graph.as_default():
        out = model.predict(np.expand_dims(encode(query), axis=0))
        print(out)
        print(np.argmax(out, axis=1))
        out = np.argmax(out, axis=1)
    
    return jsonify({'AdaBoost': ada.tolist(),
                    'BernoulliNB': ber.tolist(),
                    'DecisionTree': dt.tolist(),
                    'GradientBoost': gb.tolist(),
                    'KNNeighbors': knnp.tolist(),
                    'RandomForest': rf.tolist(),
                    'MultinomialNB': mnb.tolist(),
                    'SVM': svm.tolist(),
                    '3-layer Perceptron': out.tolist()})

if __name__ == '__main__':
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    # with open('maxent.pkl', 'rb') as f:
    #     maxent = pickle.load(f)
    with open('word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)
    model, graph = init_model()
    adaboost = joblib.load('adaboost.pkl')
    bernoulli = joblib.load('bernoullinb.pkl')
    decisiontree = joblib.load('decisiontree.pkl')
    gradientboost = joblib.load('gradientboost.pkl')
    knn = joblib.load('knn.pkl')
    randomforest = joblib.load('randomforest.pkl')
    multinomialnb = joblib.load('multinomialnb.pkl')
    svm10 = joblib.load('svm10.pkl')
    app.run(port=8080, debug=True)


