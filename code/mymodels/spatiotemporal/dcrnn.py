import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from utils.loss import *
from mymodels.spatiotemporal.dcgru_cell_tf2 import *


class DCGRU(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.model_name = f'DCGRU_Kdif{args.K_diffusion}_{args.filter_type}'
        self.P = args.P
        self.Q = args.Q
        self.D = args.D
        self.num_sensors = metadata['num_sensors']
        self.activity_embedding = args.activity_embedding
        self.timestamp_embedding = args.timestamp_embedding
        self.sensor_embedding = args.sensor_embedding
        self.K_diffusion = args.K_diffusion
        self.filter_type = args.filter_type


        self.graph_type = args.graph_type
        if self.graph_type != 'none':
            self.adj_mx = metadata['adj_mx']

        assert self.timestamp_embedding
        

        
    def build(self, input_shape):
        self.encoder = keras.layers.RNN(
            DCGRUCell(units=self.D, adj_mx=self.adj_mx, K_diffusion=self.K_diffusion,num_nodes=self.num_sensors,filter_type=self.filter_type),
            return_sequences=True, return_state=True, name='encoder'
        )
        self.decoder = keras.layers.RNN(
            DCGRUCell(units=self.D, adj_mx=self.adj_mx, K_diffusion=self.K_diffusion,num_nodes=self.num_sensors,filter_type=self.filter_type),
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

        timeofday = TE[:, :self.P, ...]
        timeofday = tf.cast(timeofday[..., -1], dtype=tf.float32) / (12*24)# timeofday
        timeofday = tf.expand_dims(tf.expand_dims(timeofday, -1), -1)
        timeofday = tf.tile(timeofday, [1, 1, self.num_sensors, 1])
        X = tf.concat((X, timeofday), -1)

        embX = self.input_layer(X)
        _, last_state = self.encoder(embX)
        embX, _ = self.decoder(tf.zeros((batch_size, self.Q, self.num_sensors, self.D)), initial_state=last_state)
        embX = tf.reshape(embX, (-1, self.Q, self.num_sensors, self.D))

        output = self.output_layer(embX)
        return output


