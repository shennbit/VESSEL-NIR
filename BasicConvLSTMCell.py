import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, concatenate
from tensorflow.keras.initializers import TruncatedNormal, Constant


class ConvLSTMCell():
    def __init__(self, filter_size, num_features, forget_bias=1.0, input_size=None,
                 activation=K.tanh):
        if input_size is not None:
            logging.warning("%s: The input_size parameter is deprecated.", self)
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._activation = activation

    def __call__(self, inputs, state):
        """Long short-term memory cell (LSTM)."""
        # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
        #state_c_n = np.int32(state.shape[3]/2)
        #c = state[:,:,:,0:state_c_n]
        #h = state[:,:,:,state_c_n:state_c_n*2]
        c, h = tf.split(state, 2, 3)

        concat = _conv_linear(args=[inputs, h], filter_size=self.filter_size, num_features=self.num_features * 4)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate j=g
        i, j, f, o = tf.split(concat, 4, 3)

        new_c = (c * K.sigmoid(f + self._forget_bias) + K.sigmoid(i) *
                 self._activation(j))
        new_h = self._activation(new_c) * K.sigmoid(o)

        new_state = K.concatenate([new_c, new_h], axis=3)

        return new_h, new_state


def _conv_linear(args, filter_size, num_features):
    if len(args) == 1:
        res = Conv2D(num_features, (filter_size[0], filter_size[1]), activation=None, padding='same')(args[0])
    else:
        cat = K.concatenate([args[0], args[1]], axis=3)
        #cat = concatenate([args[0], args[1]], axis=3)
        res = Conv2D(num_features, (filter_size[0], filter_size[1]), activation=None, padding='same')(cat)
    return res