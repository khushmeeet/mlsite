import numpy as np
from keras.models import load_model
import tensorflow as tf

def init_model():
    perceptron_model = load_model('3layer.h5')
    lstm_model = load_model('lstm2.h5')
    perceptron_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return perceptron_model, lstm_model, graph