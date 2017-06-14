from flask import Flask, jsonify, render_template, request, url_for, send_from_directory, redirect, flash
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import pickle
import numpy as np
import os
from load import init_model
from keras.preprocessing.sequence import pad_sequences

global pmodel, lmodel, graph, vectorizer

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        try:
            vector.append(word2index[word])
        except Exception:
            vector.append(0)
    padded_seq = pad_sequences([vector], maxlen=100, value=0.)
    return padded_seq

def word_feats(text):
    return dict([(word, True) for word in text.split(' ')])

def open_pkl(str):
    with open(str, 'rb') as f:
        x = pickle.load(f)
    return x

def predictor(query):
    clean_query = clean(query)
    ada = adaboost.predict(clean_query)
    ber = bernoulli.predict(clean_query)
    # nb = naivebayes.classify(word_feats(query))
    lg = logistic.predict(clean_query)
    dt = decisiontree.predict(clean_query)
    gb = gradientboost.predict(clean_query.toarray())
    knnp = knn.predict(clean_query)
    rf = randomforest.predict(clean_query)
    mnb = multinomialnb.predict(clean_query)
    svm = svm10.predict(clean_query)

    with graph.as_default():
        pout = pmodel.predict(np.expand_dims(pencode(query), axis=0))
        lout = lmodel.predict((lencode(query)))
        pout = np.argmax(pout, axis=1)
        lout = np.argmax(lout, axis=1)
    
    return {'AdaBoost': ada.tolist(),
            'BernoulliNB': ber.tolist(),
            'DecisionTree': dt.tolist(),
            'GradientBoost': gb.tolist(),
            'KNNeighbors': knnp.tolist(),
            'RandomForest': rf.tolist(),
            'MultinomialNB': mnb.tolist(),
            'MaxEnt': lg.tolist(),
            'SVM': svm.tolist(),
            '3-layer Perceptron': pout.tolist(),
            'lstm network': lout.tolist()}

# Shit, this is horrible!!
# need to find a better way to do this.
def fpredictor(query):
    clean_query = clean(query)
    ada = adaboost.predict(clean_query)
    ber = bernoulli.predict(clean_query)
    # nb = naivebayes.classify(word_feats(query))
    lg = logistic.predict(clean_query)
    dt = decisiontree.predict(clean_query)
    gb = gradientboost.predict(clean_query.toarray())
    knnp = knn.predict(clean_query)
    rf = randomforest.predict(clean_query)
    mnb = multinomialnb.predict(clean_query)
    svm = svm10.predict(clean_query)

    with graph.as_default():
        pout = pmodel.predict(np.expand_dims(pencode(query), axis=0))
        lout = lmodel.predict((lencode(query)))
        pout = np.argmax(pout, axis=1)
        lout = np.argmax(lout, axis=1)
    
    return [ada.tolist(),
            ber.tolist(),
            dt.tolist(),
            gb.tolist(),
            knnp.tolist(),
            rf.tolist(),
            mnb.tolist(),
            lg.tolist(),
            svm.tolist(),
            pout.tolist(),
            lout.tolist()]


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.mkdir(UPLOAD_FOLDER)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('fpredict', filename=filename))
    else:
        return render_template('index.html')

# the most horrible code, i have ever written
@app.route('/fpredict/<filename>', methods=['POST', 'GET'])
def fpredict(filename):
    with open('./uploads/'+filename) as f:
        text = f.read()
    text = text.split('.')
    predict_list = []
    for t in text:
        p = fpredictor(t)
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

    for i in predict_list:
        data['AdaBoost'] += i[0][0]
        data['BernoulliNB'] += i[1][0]
        data['DecisionTree'] += i[2][0]
        data['GradientBoost'] += i[3][0]
        data['KNNeighbors'] += i[4][0]
        data['RandomForest'] += i[5][0]
        data['MultinomialNB'] += i[6][0]
        data['MaxEnt'] = i[7][0]
        data['SVM'] += i[8][0]
        data['3-layer Perceptron'] += i[9][0]
        data['lstm network'] += i[10][0]
    
    # for d in data:
    #     if d != 'Naive Bayes' or d!= 'MaxEnt':
    #         print(d)
    #         data[d] /= len(text)
    #     else:
    #         pass
    return jsonify(data)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    query = request.get_data().decode('utf-8')
    data = predictor(query)
    return jsonify(data)


def main():
    global pmodel, lmodel, graph, vectorizer, naivebayes, word2index, logistic, \
    adaboost, bernoulli, decisiontree, gradientboost, knn, randomforest, multinomialnb, svm10

    with open('word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    # with open('naivebayes.pkl', 'rb') as f:
    #     naivebayes = pickle.load(f)
    pmodel, lmodel, graph = init_model()
    logistic = open_pkl('logisticreg.pkl')
    adaboost = open_pkl('adaboost.pkl')
    bernoulli = open_pkl('bernoullinb.pkl')
    decisiontree = open_pkl('decisiontree.pkl')
    gradientboost = open_pkl('gradientboost.pkl')
    knn = open_pkl('knn.pkl')
    randomforest = open_pkl('randomforest.pkl')
    multinomialnb = open_pkl('multinomialnb.pkl')
    svm10 = open_pkl('svm10.pkl')
    port = int(os.environ.get("PORT", 6000))
    app.run(host='0.0.0.0', port=port)