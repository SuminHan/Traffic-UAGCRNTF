"""
Alternative implementation of the DCGRU recurrent cell in Tensorflow 2
References
----------
Paper: https://arxiv.org/abs/1707.01926
Original implementation: https://github.com/liyaguang/DCRNN
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
# The majority of the present code originally comes from
# https://github.com/liyaguang/DCRNN/blob/master/lib/utils.py

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


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


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def build_sparse_matrix(L):
    L = L.astype('float32')
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    return tf.sparse.reorder(L)

class DCGRUCell(keras.layers.Layer):
    def __init__(self, units, adj_mx, K_diffusion, num_nodes, filter_type, **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.K_diffusion = K_diffusion
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(DCGRUCell, self).__init__(**kwargs)
        self.supports = []
        supports = []
        # the formula describing the diffsuion convolution operation in the paper
        # corresponds to the filter "dual_random_walk"
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":           
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(sp.csr_matrix(adj_mx))
        for support in supports:
            self.supports.append(build_sparse_matrix(support))
            sup0 = support
            for k in range(2, self.K_diffusion + 1):
                sup0 = support.dot(sup0)                  # (original paper version)
                # sup0 = 2 * support.dot(sup0) - sup0     # (author's repository version)
                self.supports.append(build_sparse_matrix(sup0))

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
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        for support in self.supports:
            # premultiply the concatened inputs and state with support matrices
            x_support = tf.sparse.sparse_dense_matmul(support, x0)
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)



class GTS_GCGRUCell(keras.layers.Layer):
    def __init__(self, units, num_nodes, K_diffusion, **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.num_nodes = num_nodes
        self.K_diffusion = K_diffusion
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(GTS_GCGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        print('input_shape', input_shape)
        input_shape, supports_shape = input_shape[0], input_shape[1]
        self.num_mx = 1 + self.K_diffusion
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        inputs, supports = inputs
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, supports, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, supports, h_prev, 'update'))
        print('r.shape', r.shape, 'h_prev.shape', h_prev.shape)
        c = self.activation(self.diff_conv(inputs, supports, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]


    def diff_conv(self, inputs, supports, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        print('supports.shape', supports.shape)

        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        adj_mx = supports[0, ..., 0]

        print('adj_mx.shape', adj_mx.shape)
        # for j in range(supports.shape[-1]):
        #     support = supports[0, ..., j]
        #     x_support = support @ x0
        #     x_support = tf.expand_dims(x_support, 0)
        #     # concatenate convolved signal
        #     x = tf.concat([x, x_support], axis=0)
        
        x1 = adj_mx @ x0
        _x1 = tf.expand_dims(x1, 0)
        x = tf.concat([x, _x1], axis=0)
        
        for k in range(2, self.K_diffusion+1):
            x2 = 2 * (adj_mx @ x1) - x0
            _x2 = tf.expand_dims(x2, 0)
            x = tf.concat([x, _x2], axis=0)
            x1, x0 = x2, x1

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)



class GCGRUCell(keras.layers.Layer):
    def __init__(self, units, num_nodes, filter_type, **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(GCGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        print('input_shape', input_shape)
        input_shape, supports_shape = input_shape[0], input_shape[1]
        self.num_mx = 2+1
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        inputs, supports = inputs
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, supports, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, supports, h_prev, 'update'))
        print('r.shape', r.shape, 'h_prev.shape', h_prev.shape)
        c = self.activation(self.diff_conv(inputs, supports, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]


    def diff_conv(self, inputs, supports, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        print('supports.shape', supports.shape)

        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        for j in range(supports.shape[-1]):
            support = supports[0, ..., j]
            x_support = support @ x0
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)



class GCGRUCell2(keras.layers.Layer):
    def __init__(self, units, num_nodes, filter_type, **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(GCGRUCell2, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        input_shape, supports_shape = input_shape[0], input_shape[1]
        print(supports_shape[-1])
        self.num_mx = supports_shape[-1]+1
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        inputs, supports = inputs
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, supports, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, supports, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, supports, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]


    def diff_conv(self, inputs, supports, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """

        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x0 = inputs_and_state # batch_size, num_nodes, input_dim + units

        # x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        # x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        # x = tf.expand_dims(x0, axis=0)
        xs = [x0]
        
        for j in range(supports.shape[-1]):
            support = supports[..., j]

            # premultiply the concatened inputs and state with support matrices
            
            # support = tf.sparse.from_dense(support)
            # x_support = tf.sparse.sparse_dense_matmul(support, x0)
            x_support = support @ x0
            # concatenate convolved signal
            # x = tf.concat([x, x_support], axis=0)
            xs.append(x_support)

        x = tf.stack(xs, 0)
        x = tf.reshape(x, shape=[self.num_mx, -1, self.num_nodes, input_size])
        x = tf.transpose(x, perm=[1, 2, 3, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        # x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        # x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        # x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)



class MGCGRUCell(keras.layers.Layer):
    def __init__(self, units, supports, num_nodes, **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(MGCGRUCell, self).__init__(**kwargs)
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
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        for support in self.supports:
            # premultiply the concatened inputs and state with support matrices
            x_support = tf.sparse.sparse_dense_matmul(support, x0)
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)
