import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
import importlib 
import numpy as np
from mymodels.spatiotemporal.dcgru_cell_tf2 import *


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))
    

class MGCN(keras.layers.Layer):
    def __init__(self, units, num_nodes, supports, **kwargs):
        super(MGCN, self).__init__(**kwargs)
        self.units = units
        self.num_nodes = num_nodes
        self.supports = supports

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        self.num_mx = 1 + len(self.supports)
        self.input_dim = input_shape[-1]
        self.rows_kernel = input_shape[-1] * self.num_mx

        self.kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='kernel')
        self.bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='bias')
        self.built = True


    def call(self, input):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes, units)
        """

        assert input.get_shape()[1] == self.num_nodes
        assert input.get_shape()[2] == self.input_dim

        x = input
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        for support in self.supports:
            # premultiply the concatened inputs and state with support matrices
            x_support = tf.sparse.sparse_dense_matmul(support, x0)
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, self.input_dim, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, self.input_dim * self.num_mx])
        x = tf.matmul(x, self.kernel)
        x = tf.nn.bias_add(x, self.bias)
        x = tf.reshape(x, [-1, self.num_nodes, self.units])
        return x # (batch_size, num_nodes, units)

        
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

#   def compute_mask(self, *args, **kwargs):
#     return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    # x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
   
    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x



class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x




class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x



class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x



class GCEncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, supports, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)
    self.num_sensors = supports[0].shape[0]
    self.mgcn = MGCN(d_model, self.num_sensors, supports)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)

    num_seq = x.shape[1]
    hidden_dim = x.shape[-1]
    x = tf.reshape(x, (-1, self.num_sensors, num_seq, hidden_dim))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (-1, self.num_sensors, hidden_dim))
    x = self.mgcn(x)
    x = tf.reshape(x, (-1, num_seq, self.num_sensors, hidden_dim))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (-1, num_seq, hidden_dim))

    return x


class GCEncoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, supports, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        GCEncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     supports=supports,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)
      

    return x  # Shape `(batch_size, seq_len, d_model)`.





class GCDecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               supports,
               dropout_rate=0.1):
    super(GCDecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    
    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)
    self.num_sensors = supports[0].shape[0]
    self.mgcn = MGCN(d_model, self.num_sensors, supports)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

    num_seq = x.shape[1]
    hidden_dim = x.shape[-1]
    x = tf.reshape(x, (-1, self.num_sensors, num_seq, hidden_dim))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (-1, self.num_sensors, hidden_dim))
    x = self.mgcn(x)
    x = tf.reshape(x, (-1, num_seq, self.num_sensors, hidden_dim))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (-1, num_seq, hidden_dim))

    return x



class GCDecoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               supports, dropout_rate=0.1):
    super(GCDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        GCDecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, supports=supports, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x




class GCTransformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, supports, dropout_rate=0.1):
    super().__init__()
    self.encoder = GCEncoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           supports=supports,
                           dropout_rate=dropout_rate)

    self.decoder = GCDecoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           supports=supports,
                           dropout_rate=dropout_rate)

    # self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    # logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
    logits = x

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits



class MyUAGCTransformer(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.model_name = 'MyUAGCTransformer'
        self.P = args.P
        self.Q = args.Q
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
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
            load_adj_mx = metadata['adj_mx']
            for a in [load_adj_mx.copy(), load_adj_mx.copy().T]:
                self.adj_mxs.append(build_sparse_matrix(calculate_random_walk_matrix(a).T))
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "P": self.P,
            "Q": self.Q,
            "K": self.K,
            "D": self.D,
            "teacher": self.teacher,
        })
        return config
        
    def build(self, input_shape):
        num_layers = self.L
        d_model = self.D
        dff = self.D
        num_heads = self.K
        dropout_rate = 0.1

        self.gctransformer = GCTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=-1, # do not use
            target_vocab_size=-1, # do not use
            supports=self.adj_mxs,
            dropout_rate=dropout_rate)


        if self.activity_embedding or self.timestamp_embedding:
            self.te_embed_layer = keras.models.Sequential([layers.Dense(self.D, activation='relu'),
                                                                layers.Dense(self.D),
                                                                layers.Normalization(-1)])
        if self.sensor_embedding:
            self.se_embed_layer = layers.Embedding(self.num_sensors, self.D)

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

        
        decoder_output = self.gctransformer((embX+ISTE_P, ISTE_Q))
        
        embX = tf.reshape(decoder_output, (-1, self.num_sensors, self.Q, self.D))
        embX = tf.transpose(embX, (0, 2, 1, 3))
        output = self.output_layer(embX)
        return output




