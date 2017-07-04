import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from collections import Counter
import resource
import tweepy
import boto3.session
import _pickle


session = boto3.session.Session(region_name='ap-south-1')
s3client = session.client('s3', config=boto3.session.Config(signature_version='s3v4'),
                          aws_access_key_id='AKIAJMHZHAQ2NOEWDWVA',
                          aws_secret_access_key='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')

print(session, s3client)


def most_common(lst):
    return max(set(lst), key=lst.count)


def load_from_s3(str):
    response = s3client.get_object(Bucket='mlsite-bucket', Key=str)
    body = response['Body'].read()
    if '.h5' in str:
        print('type', type(body))
        detector = load_model(body)
    else:
        detector = _pickle.loads(body)
    return detector


# def open_pkl(str):
#     with open(str, 'rb') as f:
#         x = pickle.load(f)
#     return x


word2index = load_from_s3('models/word2index.pkl')
vectorizer = load_from_s3('models/vectorizer.pkl')


def init_model():
    perceptron_model = load_model('app/static/models/3layer.h5')
    lstm_model = load_model('app/static/models/lstm.h5')
    cnn_model = load_model('app/static/models/cnn.h5')
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    perceptron_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return perceptron_model, lstm_model, cnn_model, graph


pmodel, lmodel, cnn, graph = init_model()
logistic = load_from_s3('models/logisticreg.pkl')
adaboost = load_from_s3('models/adaboost.pkl')
bernoulli = load_from_s3('models/bernoullinb.pkl')
decisiontree = load_from_s3('models/decisiontree.pkl')
gradientboost = load_from_s3('models/gradientboost.pkl')
knn = load_from_s3('models/knn.pkl')
randomforest = load_from_s3('models/randomforest.pkl')
multinomialnb = load_from_s3('models/multinomialnb.pkl')
svm10 = load_from_s3('models/svm10.pkl')

auth = tweepy.OAuthHandler('hXJ8TwQzVya3yYwQN1GNvGNNp', 'diX9CFVOOfWNli2KTAYY13vZVJgw1sYlEeOTxsLsEb2x73oI8S')
auth.set_access_token('2155329456-53H1M9QKqlQbEkLExgVgkeallweZ9N74Aigm9Kh',
                      'waDPwamuPkYHFLdVNZ5YF2SNWuYfGHDVFue6bEbEGjTZb')

api = tweepy.API(auth)


def clean(query):
    return vectorizer.transform([query])


def pencode(text):
    vector = np.zeros(9451)
    for i, word in enumerate(text.split(' ')):
        try:
            vector[word2index[word]] = 1
        except KeyError:
            vector[i] = 0
    return vector


def lencode(text):
    vector = []
    for word in text.split(' '):
        try:
            vector.append(word2index[word])
        except KeyError:
            vector.append(0)
    padded_seq = pad_sequences([vector], maxlen=100, value=0.)
    return padded_seq


def word_feats(text):
    return dict([(word, True) for word in text.split(' ')])


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
        cnn_out = cnn.predict(lencode(query))
        pout = np.argmax(pout, axis=1)
        lout = np.argmax(lout, axis=1)
        cnn_out = np.argmax(cnn_out, axis=1)

    print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

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
            lout.tolist()[0],
            cnn_out.tolist()[0]]


def get_most_count(x):
    return Counter(x).most_common()[0][0]


def processing_results(query):
    predict_list = []
    line_sentiment = []
    for t in query:
        p = predictor(t)
        line_sentiment.append(most_common(p))
        predict_list.append(p)

    data = {'AdaBoost': 0,
            'BernoulliNB': 0,
            'DecisionTree': 0,
            'GradientBoost': 0,
            'KNNeighbors': 0,
            'RandomForest': 0,
            'MultinomialNB': 0,
            'Logistic Regression': 0,
            'SVM': 0,
            '3-layer Perceptron': 0,
            'LSTM network': 0,
            'Convolutional Neural Network': 0}

    # overal per sentence
    predict_list = np.array(predict_list)
    i = 0
    for key in data:

        data[key] = get_most_count(predict_list[:, i])
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

    # overall score
    score = most_common(list(data.values()))

    print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return data, emotion_sents, score, line_sentiment, query

