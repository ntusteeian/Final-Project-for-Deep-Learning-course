import tensorflow as tf
from preprocessing import *


NUM_CLASS = 21

class unet():

    def __init__(self, NUM_CLASS, is_train):

        self.num_class = NUM_CLASS
        self.is_train = is_train

    def model(self, input):

        num_dlayer, num_uplayer = 14, 13
        base_filter = 32
        layer_input = input


        d_conv, d_act, d_batch = [0] * num_dlayer, [0] * num_dlayer, [0] * num_dlayer
        for layer in range(1, num_dlayer+1):
           if (layer % 3) != 0 :
                d_conv[layer-1] = tf.layers.conv2d(layer_input, filters = base_filter, kernel_size = [3, 3],
                                                   strides = (1, 1), padding = 'same', name = 'd_conv{}'.format(layer))
                # d_batch[layer-1] = tf.layers.batch_normalization(d_conv[layer-1], training = self.is_train, name = 'd_batch{}'.format(layer))
                d_act[layer-1] = tf.nn.relu(d_conv[layer-1], name = 'd_act{}'.format(layer))
                layer_input = d_act[layer-1]
           else :
                base_filter = base_filter * 2
                pool = tf.layers.max_pooling2d(layer_input, pool_size = [2, 2], strides = [2, 2], padding = 'same', name = 'pool{}'.format(layer))
                layer_input = pool

        print (d_conv)
        print ('------------------------------------')

        trans_conv, up_conv, up_act, concat = [0] * num_dlayer, [0] * num_dlayer, [0] * num_dlayer, [0] * num_dlayer
        # up_batch = [0] * num_dlayer
        concat_index = num_uplayer - 3

        for layer in range(0, num_uplayer):

            if layer != (num_uplayer - 1):

                if layer % 3 == 0:
                    base_filter = base_filter // 2
                    trans_conv[layer] = tf.layers.conv2d_transpose(layer_input, filters = base_filter, kernel_size = [3, 3],
                                                                   strides = (2, 2), padding = 'same', name = 'trans_conv{}'.format(layer+1))
                    # up_batch[layer] = tf.layers.batch_normalization(trans_conv[layer], training=self.is_train, name='up_batch{}'.format(layer+1))
                    up_act[layer] = tf.nn.relu(trans_conv[layer], name = 'up_act{}'.format(layer+1))

                    concat[layer] = tf.concat([d_conv[concat_index], up_act[layer]], axis = 3, name = 'concat{}'.format(layer+1))
                    layer_input = concat[layer]
                    concat_index -= 3
                else:
                    up_conv[layer] = tf.layers.conv2d(layer_input, filters = base_filter, kernel_size=[3, 3],
                                                       strides=(1, 1), padding='same', name='up_conv{}'.format(layer+1))
                    up_act[layer] = tf.nn.relu(up_conv[layer], name = 'up_act{}'.format(layer+1))
                    layer_input = up_act[layer]

            elif layer == (num_uplayer - 1):

                logits = tf.layers.conv2d(layer_input, filters = self.num_class, kernel_size=[1, 1],
                                                  strides=(1, 1), padding='same', name='up_conv{}'.format(layer + 1))
                softmax_output = tf.nn.softmax(logits, name = 'softmax_output')


        return logits, softmax_output

