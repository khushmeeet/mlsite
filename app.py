from flask import Flask, jsonify, render_template, request, url_for
from sklearn.externals import joblib
import pickle
import numpy as np
import os
from load import init_model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

global pmodel, lmodel, graph

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


def pencode(text):
    vector = np.zeros(9451)
    for i, word in enumerate(text.split(' ')):
        try:
            vector[word2index[word]] = 1
        except Exception:
            vector[i] = 0
    return vector

def lencode(text):
    vector = []
    for word in text.split(' '):
        vector.append(word2index[word])
    padded_seq = pad_sequences([vector], maxlen=100, value=0.)
    return padded_seq

def word_feats(text):
    return dict([(word, True) for word in text.split(' ')])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    query = request.get_data().decode('utf-8')
    clean_query = clean(query)
    ada = adaboost.predict(clean_query)
    ber = bernoulli.predict(clean_query)
    nb = naivebayes.classify(word_feats(query))
    me = maxent.classify(word_feats(query))
    dt = decisiontree.predict(clean_query)
    gb = gradientboost.predict(clean_query.toarray())
    knnp = knn.predict(clean_query)
    rf = randomforest.predict(clean_query)
    mnb = multinomialnb.predict(clean_query)
    svm = svm10.predict(clean_query)

    with graph.as_default():
        pout = pmodel.predict(np.expand_dims(pencode(query), axis=0))
        lout = lmodel.predict((lencode(query)))
        print(lout)
        pout = np.argmax(pout, axis=1)
        lout = np.argmax(lout, axis=1)
        print(lout)
    return jsonify({'AdaBoost': ada.tolist(),
                    'BernoulliNB': ber.tolist(),
                    'DecisionTree': dt.tolist(),
                    'GradientBoost': gb.tolist(),
                    'KNNeighbors': knnp.tolist(),
                    'RandomForest': rf.tolist(),
                    'MultinomialNB': mnb.tolist(),
                    'Naive Bayes': nb,
                    'MaxEnt': me,
                    'SVM': svm.tolist(),
                    '3-layer Perceptron': pout.tolist(),
                    'lstm network': lout.tolist()})

if __name__ == '__main__':
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('naivebayes.pkl', 'rb') as f:
        naivebayes = pickle.load(f)
    with open('word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)
    with open('maxent.pkl', 'rb') as f:
        maxent = pickle.load(f)
    pmodel, lmodel, graph = init_model()
    adaboost = joblib.load('adaboost.pkl')
    bernoulli = joblib.load('bernoullinb.pkl')
    decisiontree = joblib.load('decisiontree.pkl')
    gradientboost = joblib.load('gradientboost.pkl')
    knn = joblib.load('knn.pkl')
    randomforest = joblib.load('randomforest.pkl')
    multinomialnb = joblib.load('multinomialnb.pkl')
    svm10 = joblib.load('svm10.pkl')
    app.run(port=8080, debug=True)