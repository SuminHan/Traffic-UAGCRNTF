import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from utils.loss import *
from mymodels.spatiotemporal.dcgru_cell_tf2 import *



class MyUAGCRN(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.model_name = f'MyUAGCRN'
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
        if self.graph_type != 'none':
            adj_mx = metadata['adj_mx']
            self.adj_mxs.append(build_sparse_matrix(calculate_random_walk_matrix(adj_mx).T))
            self.adj_mxs.append(build_sparse_matrix(calculate_random_walk_matrix(adj_mx.T).T))

        
    def build(self, input_shape):
        if self.activity_embedding or self.timestamp_embedding:
            self.te_embed_layer = keras.models.Sequential([layers.Dense(self.D, activation='relu'),
                                                                layers.Dense(self.D),
                                                                layers.Normalization(-1)])
        if self.sensor_embedding:
            self.se_embed_layer = layers.Embedding(self.num_sensors, self.D)

        self.encoder = keras.layers.RNN(
            MGCGRUCell(units=self.D, supports=self.adj_mxs, num_nodes=self.num_sensors),
            return_sequences=True, return_state=True, name='encoder'
        )
        self.decoder = keras.layers.RNN(
            MGCGRUCell(units=self.D, supports=self.adj_mxs, num_nodes=self.num_sensors),
            return_sequences=True, return_state=True, name='decoder'
        )

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
        
        # ISTE_P = tf.transpose(STE_P, (0, 2, 1, 3))
        # ISTE_P = tf.reshape(ISTE_P, (-1, self.P, self.D))
        # ISTE_Q = tf.transpose(STE_Q, (0, 2, 1, 3))
        # ISTE_Q = tf.reshape(ISTE_Q, (-1, self.Q, self.D))

        embX = self.input_layer(X)
        _, last_state = self.encoder(embX + STE_P)
        embX, _ = self.decoder(STE_Q, initial_state=last_state)
        embX = tf.reshape(embX, (-1, self.Q, self.num_sensors, self.D))

        output = self.output_layer(embX)
        return output
