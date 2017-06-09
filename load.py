import numpy as np
from keras.models import load_model
import tensorflow as tf

def init_model():
    model = load_model('3layer.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return model, graph