import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
from collections import Counter


def most_common(lst):
    return max(set(lst), key=lst.count)


def open_pkl(str):
    with open(str, 'rb') as f:
        x = pickle.load(f)
    return x


word2index = open_pkl('app/static/models/word2index.pkl')
vectorizer = open_pkl('app/static/models/vectorizer.pkl')


def init_model():
    perceptron_model = load_model('app/static/models/3layer.h5')
    lstm_model = load_model('app/static/models/lstm.h5')
    perceptron_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return perceptron_model, lstm_model, graph


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


# def predictor(query):
#     clean_query = clean(query)
#     ada = adaboost.predict(clean_query)
#     ber = bernoulli.predict(clean_query)
#     lg = logistic.predict(clean_query)
#     dt = decisiontree.predict(clean_query)
#     gb = gradientboost.predict(clean_query.toarray())
#     knnp = knn.predict(clean_query)
#     rf = randomforest.predict(clean_query)
#     mnb = multinomialnb.predict(clean_query)
#     svm = svm10.predict(clean_query)
#
#     with graph.as_default():
#         pout = pmodel.predict(np.expand_dims(pencode(query), axis=0))
#         lout = lmodel.predict((lencode(query)))
#         pout = np.argmax(pout, axis=1)
#         lout = np.argmax(lout, axis=1)
#
#     return {'AdaBoost': ada.tolist(),
#             'BernoulliNB': ber.tolist(),
#             'DecisionTree': dt.tolist(),
#             'GradientBoost': gb.tolist(),
#             'KNNeighbors': knnp.tolist(),
#             'RandomForest': rf.tolist(),
#             'MultinomialNB': mnb.tolist(),
#             'MaxEnt': lg.tolist(),
#             'SVM': svm.tolist(),
#             '3-layer Perceptron': pout.tolist(),
#             'lstm network': lout.tolist()}


def predictor(query):
    clean_query = clean(query)
    ada = adaboost.predict(clean_query)
    ber = bernoulli.predict(clean_query)
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

    return [ada.tolist()[0],
            ber.tolist()[0],
            dt.tolist()[0],
            gb.tolist()[0],
            knnp.tolist()[0],
            rf.tolist()[0],
            mnb.tolist()[0],
            lg.tolist()[0],
            svm.tolist()[0],
            pout.tolist()[0],
            lout.tolist()[0]]


def get_most_count(x):
    return Counter(x).most_common()[0][0]


pmodel, lmodel, graph = init_model()
logistic = open_pkl('app/static/models/logisticreg.pkl')
adaboost = open_pkl('app/static/models/adaboost.pkl')
bernoulli = open_pkl('app/static/models/bernoullinb.pkl')
decisiontree = open_pkl('app/static/models/decisiontree.pkl')
gradientboost = open_pkl('app/static/models/gradientboost.pkl')
knn = open_pkl('app/static/models/knn.pkl')
randomforest = open_pkl('app/static/models/randomforest.pkl')
multinomialnb = open_pkl('app/static/models/multinomialnb.pkl')
svm10 = open_pkl('app/static/models/svm10.pkl')