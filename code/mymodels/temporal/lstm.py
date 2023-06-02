import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from utils.loss import *


class MyARLSTM(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.model_name = f'MyARLSTM'
        self.P = args.P
        self.Q = args.Q
        self.D = args.D
        self.num_sensors = metadata['num_sensors']
        self.activity_embedding = args.activity_embedding
        self.timestamp_embedding = args.timestamp_embedding
        self.sensor_embedding = args.sensor_embedding
        self.sensor_node2vec = args.sensor_node2vec
        if self.sensor_node2vec:
            self.SE = metadata['SEN2V']

        self.sensor_encoding = np.arange(self.num_sensors)
        self.adj_mxs = []

        self.graph_type = args.graph_type
        assert self.graph_type == 'none'
        if self.graph_type != 'none':
            load_adj_mx = metadata['adj_mx']
            for a in [load_adj_mx.copy(), load_adj_mx.copy().T]:
                np.fill_diagonal(a, 0)
                row_sums = a.sum(axis=1)
                adj_mx = a / (row_sums[:, np.newaxis] + 1e-10)
                self.adj_mxs.append(build_sparse_matrix(sp.csr_matrix(adj_mx)))

        
    def build(self, input_shape):
        if self.activity_embedding or self.timestamp_embedding:
            self.te_embed_layer = keras.models.Sequential([layers.Dense(self.D, activation='relu'),
                                                                layers.Dense(self.D),
                                                                layers.Normalization(-1)])
        if self.sensor_embedding:
            self.se_embed_layer = layers.Embedding(self.num_sensors, self.D)

        self.encoder = LSTM(self.D, return_sequences=True, return_state=True, name='encoder')
        self.decoder = LSTM(self.D, return_sequences=True, return_state=True, name='decoder')

        self.input_layer = keras.models.Sequential([layers.Dense(self.D, activation='relu'),
                                                            layers.Dense(self.D)])
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, X, TE):
        # X.shape (batch_size, P, N, 1)
        # TE.shape (batch_size, P+Q, TE_channel)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]

        # Activity embedding
        if self.activity_embedding:
            TE = tf.cast(TE, dtype=tf.float32)
            TE = tf.expand_dims(TE, -2)
            TE = tf.tile(TE, [1, 1, self.num_sensors, 1])
            TE = self.te_embed_layer(TE)
        elif self.timestamp_embedding:
            weekday = tf.one_hot(tf.cast(TE[..., 0], dtype=tf.int32), depth=7)
            timeofday = tf.one_hot(tf.cast(TE[..., 1], dtype=tf.int32), depth=12*24)
            TE = tf.cast(tf.concat((weekday, timeofday), -1), dtype=tf.float32)
            TE = tf.expand_dims(TE, -2)
            TE = tf.tile(TE, [1, 1, self.num_sensors, 1])
            TE = self.te_embed_layer(TE)
        else:
            TE = tf.zeros((batch_size, self.P+self.Q, self.num_sensors, self.D))


        # Sensor embedding
        if self.sensor_embedding:
            SE = self.se_embed_layer(self.sensor_encoding)
            SE = tf.expand_dims(tf.expand_dims(SE, 0), 0)
            SE = tf.tile(SE, [batch_size, self.P + self.Q, 1, 1])
        elif self.sensor_node2vec:
            SE = self.SE
            SE = tf.expand_dims(tf.expand_dims(SE, 0), 0)
            SE = tf.tile(SE, [batch_size, self.P + self.Q, 1, 1])
        else:
            SE = tf.zeros((batch_size, self.P+self.Q, self.num_sensors, self.D))
        
        
        STE = TE + SE # (-1, self.P+self.Q, self.num_sensors, self.D)
        STE_P, STE_Q = STE[:, :self.P, ...], STE[:, self.P:, ...]
        
        ISTE_P = tf.transpose(STE_P, (0, 2, 1, 3))
        ISTE_P = tf.reshape(ISTE_P, (-1, self.P, self.D))
        ISTE_Q = tf.transpose(STE_Q, (0, 2, 1, 3))
        ISTE_Q = tf.reshape(ISTE_Q, (-1, self.Q, self.D))

        embX = self.input_layer(X)
        embX = tf.transpose(embX, (0, 2, 1, 3))
        embX = tf.reshape(embX, (-1, self.P, self.D))


        _, state_h, state_c = self.encoder(embX + ISTE_P)
        embX, _, _ = self.decoder(ISTE_Q, initial_state=(state_h, state_c))
        
        embX = tf.reshape(embX, (-1, self.num_sensors, self.Q, self.D))
        embX = tf.transpose(embX, (0, 2, 1, 3))
        output = self.output_layer(embX)
        return output


