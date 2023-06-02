import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

def build_sparse_matrix(L):
    L = L.astype('float32')
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    return tf.sparse.reorder(L)

import scipy.sparse as sp
from scipy.sparse import linalg


class GMAN(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = 8
        self.d = 8
        self.D = args.D
        self.L = args.L
        self.num_sensors = metadata['num_sensors']
        self.model_name = f'GMAN'
        self.activity_embedding = args.activity_embedding
        self.sensor_embedding = args.sensor_embedding

        self.graph_type = args.graph_type
        self.pos_encoding = np.arange(self.num_sensors)
        self.sensor_node2vec = args.sensor_node2vec
        if self.sensor_node2vec:
            self.SE = metadata['SEN2V']

        self.num_encoder_layers = self.L
        self.num_decoder_layers = self.L
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "P": self.P,
            "Q": self.Q,
            "K": self.K,
            "D": self.D,
        })
        return config
        
    def build(self, input_shape):
        
        
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.num_encoder_layers)]
        self.trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.num_decoder_layers)]

        self.se_embed_layer = layers.Embedding(self.num_sensors, self.D)
        self.te_embed_layer = keras.models.Sequential([layers.Dense(self.D, activation='relu', use_bias=False),
                                                            layers.Dense(self.D, use_bias=False)])

        self.input_layer = keras.models.Sequential([layers.Dense(self.D, activation='relu'),
                                                            layers.Dense(self.D)])

        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])

        self.config = self.get_config()
    
    def call(self, X, TE):
        # X.shape (batch_size, P, N, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]


        if self.activity_embedding:
            TE = tf.cast(TE, dtype=tf.float32)
            activityTE = TE#[..., :3] # (batch, P, 5)
            activityTE = tf.expand_dims(activityTE, -2)
            activityTE = tf.tile(activityTE, [1, 1, self.num_sensors, 1]) # (batch, P, num_sensors, 5)
            TE = activityTE
            # sensorTE = TE[..., 3:] # (batch, P, num_sensors)
            # sensorTE = tf.expand_dims(sensorTE, -1) # (batch, P, num_sensors, 1)
            # TE = tf.concat((activityTE, sensorTE), -1)
            
        else:
            weekday = tf.one_hot(tf.cast(TE[..., 0], dtype=tf.int32), depth=7)
            minofday = tf.one_hot(tf.cast(TE[..., 1], dtype=tf.int32), depth=12*24)
            # timeofday = tf.expand_dims(TE[..., 1], -1)
            TE = tf.cast(tf.concat((weekday, minofday), -1), dtype=tf.float32)
            TE = tf.expand_dims(TE, -2)
            TE = tf.tile(TE, [1, 1, self.num_sensors, 1])
            print('TE.layer', TE.shape)

        TE = self.te_embed_layer(TE)

        if self.sensor_embedding:
            SE = self.se_embed_layer(self.pos_encoding)
            SE = tf.expand_dims(tf.expand_dims(SE, 0), 0)
            SE = tf.tile(SE, [batch_size, self.P + self.Q, 1, 1])
        elif self.sensor_node2vec:
            SE = self.SE
            SE = tf.expand_dims(tf.expand_dims(SE, 0), 0)
            SE = tf.tile(SE, [batch_size, self.P + self.Q, 1, 1])
        else:
            SE = tf.zeros_like(TE)
        print('TE.layer', TE.shape)
        print('SE.layer', SE.shape)

        # TE = tf.tile(TE, [1, 1, self.num_sensors, 1])
        
        STE = TE + SE # (-1, self.P+self.Q, self.num_sensors, self.D)
        STE_P, STE_Q = STE[:, :self.P, ...], STE[:, self.P:, ...]
        print('STE_P.layer', STE_P.shape)
        print('STE_Q.layer', STE_Q.shape)
        
        
        print('X.shape', X.shape)
        embX = self.input_layer(X)
        print('embX.shape', embX.shape)
        
        for i in range(self.num_encoder_layers):
            embX = self.GSTA_enc[i](embX, STE_P)
        
        embX = self.trans_layer(embX, STE_P, STE_Q)

        for i in range(self.num_decoder_layers):
            embX = self.GSTA_dec[i](embX, STE_Q)

        # (batch_size*num_sensors, Q, D)
        print('final embX.shape', embX.shape)
        decoder_output = self.output_layer(embX)

        return decoder_output



class SpatialMaskAttention(tf.keras.layers.Layer):
    def __init__(self, K, d, adj_mx):
        super(SpatialMaskAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.adj_mx = adj_mx

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        # X = X + STE
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        
        # mask
        batch_size = tf.shape(X)[0]
        num_step = tf.shape(X)[1]
        N = tf.shape(X)[2]
        mask = self.adj_mx > 0
        mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
        mask = tf.tile(mask, multiples = (K * batch_size, num_step, 1, 1))
        mask = tf.cast(mask, dtype = tf.bool)
        attention = tf.compat.v2.where(
            condition = mask, x = attention, y = -2 ** 15 + 1)

        attention = tf.nn.softmax(attention, axis = -1)

        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, K, d, use_mask=True):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.use_mask = use_mask

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        if self.use_mask:
            batch_size = tf.shape(X)[0]
            num_step = tf.shape(X)[1]
            N = tf.shape(X)[2]
            mask = tf.ones(shape = (num_step, num_step))
            mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
            mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
            mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
            mask = tf.cast(mask, dtype = tf.bool)
            attention = tf.compat.v2.where(
                condition = mask, x = attention, y = -2 ** 15 + 1)
            
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
    
class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.D = D

    def build(self, input_shape):
        self.FC_S = keras.Sequential([
            layers.Dense(self.D, use_bias=False),])
        self.FC_T = keras.Sequential([
            layers.Dense(self.D),])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.D),])
        
    def call(self, HS, HT):
        XS = self.FC_S(HS)
        XT = self.FC_T(HT)
        
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.FC_H(H)
        return H
    

class GSTAttBlock(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(GSTAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.SA_layer = SpatialAttention(self.K, self.d)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        HS = self.SA_layer(X, STE)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        # H = HS + HT
        return X + H
    

class TransformAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(TransformAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D)])
        
    def call(self, X, STE_P, STE_Q):
        K = self.K
        d = self.d
        D = self.D
        
        query = self.FC_Q(STE_Q)
        key = self.FC_K(STE_P)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))   
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X