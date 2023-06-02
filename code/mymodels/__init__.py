# from mymodels.basic import *
# from mymodels.lstm import *
# from mymodels.lstm_shift import *
# from mymodels.fclstm import *
# from mymodels.gman import *
# from mymodels.temporal_transformer import *
# from mymodels.rawlstm import *
# from mymodels.convlstm import *
# from mymodels.conv_and_lstm import *
# from mymodels.lstm_stoken import *
# from mymodels.cnn_stacks import *
# from mymodels.random_walk import *
# from mymodels.random_walk_plus import *

from mymodels import *
from mymodels.temporal import *
from mymodels.spatiotemporal import *

import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname) 


import numpy as np
import tensorflow as tf


class LastRepeat(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.P = args.P
        self.Q = args.Q
        self.model_name = 'LastRepeat'
        
    def call(self, X, TE):
        return tf.tile(tf.expand_dims(X[:, -1, :, :], 1), [1, self.Q, 1, 1])
