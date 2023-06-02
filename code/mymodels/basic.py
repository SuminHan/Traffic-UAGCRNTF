import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from submodules.pos_encoding_2d import *
from utils import *


class LastRepeat(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(LastRepeat, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.model_name = 'LastRepeat'
    
    def call(self, X, TE):
        return tf.tile(X[:, -1:, ...], [1, self.Q, 1, 1, 1]) 
