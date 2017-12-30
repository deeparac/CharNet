import tensorflow as tf
import numpy as np
import time

class CharNet(object):
    """docstring for CharNet."""
    def __init__(self, conv_layers,
                        fc_layers,
                        l0,
                        alphabet_size,
                        encoder,
                        **args
    ):
        super(CharNet, self).__init__()
        tf.set_random_seed(time.time())
        self.l0 = l0
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.alphabet_size = alphabet_size

        initializer = tf.contrib.layers.xavier_initializer()

        with tf.name_scope('Input'):
            self.input_num = tf.placeholder(tf.float32, shape=[None, 6],
                                           name='input_num')
            self.input_x = tf.placeholder(tf.int64, shape=[None, self.l0],
                                          name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape=[None, 1],
                                          name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                    name='dropout_keep_prob')

        with tf.name_scope('Embedding'):
            x = tf.nn.embedding_lookup(encoder, self.input_x)
            x = tf.expand_dims(x, -1)

        # Configure conv layers
        for i, layer_params in enumerate(conv_layers):
            with tf.name_scope("Convolution"):
                filter_param = [
                    layer_params[1],
                    x.get_shape()[2].value, # l0
                    x.get_shape()[3].value, # channels
                    layer_params[0]
                ]
                W = tf.Variable(initializer(filter_param), dtype='float32', name='filter')

                conv_layer = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID', name='conv')
                conv_layer = tf.nn.relu(conv_layer, name='act_relu')

            if not layer_params[-1] is None:
                with tf.name_scope("MaxPooling"):
                    pool_layer = tf.nn.max_pool(conv_layer,
                                            ksize=[1, layer_params[-1], 1, 1],
                                            strides=[1, layer_params[-1], 1, 1],
                                            padding='VALID')
                    x = tf.transpose(pool_layer, [0, 1, 3, 2])
            else:
                x = tf.transpose(conv_layer, [0, 1, 3, 2])

        # flatten conv output for fc
        with tf.name_scope("Flatten"):
            x = tf.contrib.layers.flatten(x)

        with tf.name_scope("Concat"):
            x = tf.concat([x, self.input_num], axis=1)

        # Configure fc layers
        for i, layer_units in enumerate(fc_layers):
            with tf.name_scope("FullyConnected"):
                W = tf.Variable(initializer([x.get_shape()[-1].value, layer_units]),
                                dtype='float32', name='W')
                b = tf.Variable(initializer([layer_units]),
                                dtype='float32', name='W')
                x = tf.nn.xw_plus_b(x, W, b, name='fully-connected')
                x = tf.nn.relu(x)

            with tf.name_scope("Dropout"):
                x = tf.nn.dropout(x, self.dropout_keep_prob)

        with tf.name_scope("Output"):
            W = tf.Variable(initializer([x.get_shape()[-1].value, 1]),
                            dtype='float32', name='W')
            b = tf.Variable(initializer([1]),
                            dtype='float32', name='W')
            yhat = tf.nn.xw_plus_b(x, W, b, name='output')
            self.yhat = tf.reshape(yhat, [-1])

        with tf.name_scope("Loss"):
            y = tf.reshape(self.input_y, [-1])
            self.loss = tf.keras.metrics.mean_squared_logarithmic_error(self.yhat, y)