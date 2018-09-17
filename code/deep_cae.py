""" Deep Convolutional Autoconvder
"""
from dependency import *
import utils.net_element as ne
from utils.decorator import lazy_method, lazy_method_no_scope


class DeepCAE:

    def __init__(self, 
                 # relu bounds
                 output_low_bound, 
                 output_up_bound,
                 # conv layers
                 conv_filter_size=[3,3], 
                 conv_channel_sizes=[128, 128, 128, 128, 1], #[256, 256, 256, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.4, 0.2, 0.2],
                 # deconv layers
                 decv_filter_size=[3,3], 
                 decv_channel_sizes=[1, 128, 128, 128, 128], #[1, 256, 256, 256]
                 decv_leaky_ratio=[0.2, 0.2, 0.4, 0.4, 0.4],
                 # encoder fc layers
                 enfc_state_sizes=[4096], 
                 enfc_leaky_ratio=[0.2, 0.2],
                 enfc_drop_rate=[0, 0.75],
                 # bottleneck
                 center_state_size=1024, 
                 # decoder fc layers
                 defc_state_sizes=[4096],
                 defc_leaky_ratio=[0.2, 0.2],
                 defc_drop_rate=[0.75, 0],
                 # switch
                 use_batch_norm = False
                ):
        # conv layers
        self.conv_filter_size = conv_filter_size
        self.conv_channel_sizes = conv_channel_sizes
        self.conv_leaky_ratio = conv_leaky_ratio
        # deconv layers
        self.decv_filter_size = decv_filter_size 
        self.decv_channel_sizes = decv_channel_sizes
        self.decv_leaky_ratio = decv_leaky_ratio
        # encoder fc layers
        self.enfc_state_sizes = enfc_state_sizes 
        self.enfc_leaky_ratio = enfc_leaky_ratio
        self.enfc_drop_rate = enfc_drop_rate
        # bottleneck
        self.center_state_size = center_state_size
        # decoder fc layers
        self.defc_state_sizes = defc_state_sizes
        self.defc_leaky_ratio = defc_leaky_ratio
        self.defc_drop_rate = defc_drop_rate
         # relu bounds
        self.output_low_bound = output_low_bound
        self.output_up_bound = output_up_bound
        # switch
        self.use_batch_norm = use_batch_norm

        self.conv_out_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.conv_channel_sizes[-1]]
        self.decv_in_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.decv_channel_sizes[0]]

        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        self.enfc_weights, self.enfc_biases, self.num_enfc = self.enfc_weights_biases()
        self.defc_weights, self.defc_biases, self.num_defc = self.defc_weights_biases()
        self.decv_filters, self.decv_biases, self.num_decv = self.decv_weights_biases()


    @lazy_method
    def conv_weights_biases(self):
        return self._conv_weights_biases("W_conv", "b_conv", self.conv_filter_size, FLAGS.NUM_CHANNELS, self.conv_channel_sizes)


    @lazy_method
    def decv_weights_biases(self):
        in_channel = self.decv_channel_sizes[0]
        channel_sizes = self.decv_channel_sizes[1:] + [FLAGS.NUM_CHANNELS]
        return self._conv_weights_biases("W_decv", "b_decv", self.decv_filter_size, in_channel, channel_sizes, True)
    

    def _conv_weights_biases(self, W_name, b_name, filter_size, in_channel, channel_sizes, transpose=False):
        num_layer = len(channel_sizes)
        _weights = {}
        _biases = {}
        for idx in range(num_layer):
            W_key = "{}{}".format(W_name, idx)
            if transpose:
                W_shape = filter_size+[channel_sizes[idx], in_channel]
            else:
                W_shape = filter_size+[in_channel, channel_sizes[idx]]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key)

            b_key = "{}{}".format(b_name, idx)
            b_shape = [channel_sizes[idx]]
            _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_channel = channel_sizes[idx]

            # tensorboard
            tf.summary.histogram("Filter_"+W_key, _weights[W_key])
            tf.summary.histogram("Bias_"+b_key, _biases[b_key])

        return _weights, _biases, num_layer


    @lazy_method
    def enfc_weights_biases(self):
        in_size = self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]
        state_sizes = self.enfc_state_sizes + [self.center_state_size]
        return self._fc_weights_biases("W_enfc", "b_enfc", in_size, state_sizes)

    
    @lazy_method
    def defc_weights_biases(self):
        in_size = self.center_state_size
        out_size = self.decv_in_shape[0] * self.decv_in_shape[1] * self.decv_in_shape[2]
        state_sizes = self.defc_state_sizes + [out_size]
        return self._fc_weights_biases("W_defc", "b_defc", in_size, state_sizes)

    
    def _fc_weights_biases(self, W_name, b_name, in_size, state_sizes):
        num_layer = len(state_sizes)
        _weights = {}
        _biases = {}
        for idx in range(num_layer):
            W_key = "{}{}".format(W_name, idx)
            W_shape = [in_size, state_sizes[idx]]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key)

            b_key = "{}{}".format(b_name, idx)
            b_shape = [state_sizes[idx]]
            _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_size = state_sizes[idx]

            # tensorboard
            tf.summary.histogram("Weight_"+W_key, _weights[W_key])
            tf.summary.histogram("Bias_"+b_key, _biases[b_key])

        return _weights, _biases, num_layer

    
    @lazy_method
    def conv_layers(self, inputs, W_name="W_conv", b_name="b_conv"):
        net = tf.reshape(inputs, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        for layer_id in range(self.num_conv):
            filter_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_filter = self.conv_filters[filter_name]
            curr_bias = self.conv_biases[bias_name]
            # convolution
            net = ne.conv2d(net, filters=curr_filter, biases=curr_bias)
            # batch normalization
            if self.use_batch_norm:
                net = ne.batch_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            net = ne.leaky_relu(net, self.conv_leaky_ratio[layer_id])
            #net = ne.max_pool_2x2(net) # Pooling

        net = tf.identity(net, name='output')
        return net


    @lazy_method
    def enfc_layers(self, inputs, W_name="W_enfc", b_name="b_enfc"):
        net = tf.reshape(inputs, [-1, self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]])
        for layer_id in range(self.num_enfc):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.enfc_weights[weight_name]
            curr_bias = self.enfc_biases[bias_name]
            net = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
            # batch normalization
            if self.use_batch_norm:
                net = ne.batch_norm(net, self.is_training, axis=1)
            #net = ne.leaky_brelu(net, self.enfc_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            net = ne.leaky_relu(net, self.enfc_leaky_ratio[layer_id])
            net = ne.drop_out(net, self.enfc_drop_rate[layer_id], self.is_training)
            #net = ne.elu(net)

        net = tf.identity(net, name='output')
        return net
    

    @lazy_method
    def defc_layers(self, inputs, W_name="W_defc", b_name="b_defc"):
        net = inputs
        for layer_id in range(self.num_enfc):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.defc_weights[weight_name]
            curr_bias = self.defc_biases[bias_name]
            net = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
            # batch normalization
            if self.use_batch_norm:
                net = ne.batch_norm(net, self.is_training, axis=1)
            
            #net = ne.leaky_brelu(net, self.defc_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            net = ne.leaky_relu(net, self.defc_leaky_ratio[layer_id])
            net = ne.drop_out(net, self.defc_drop_rate[layer_id], self.is_training)
            #net = ne.elu(net)

        net = tf.identity(net, name='output')
        net = tf.reshape(net, [-1] + self.decv_in_shape)
        return net

    
    @lazy_method
    def decv_layers(self, inputs, W_name="W_decv", b_name="b_decv"):
        net = inputs
        for layer_id in range(self.num_decv):
            filter_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_filter = self.decv_filters[filter_name]
            curr_bias = self.decv_biases[bias_name]
            # de-convolution
            net = ne.conv2d_transpose(net, filters=curr_filter, biases=curr_bias)
            # batch normalization
            if self.use_batch_norm:
                net = ne.batch_norm(net, self.is_training)
            if layer_id == self.num_decv-1: # last layer
                net = ne.leaky_brelu(net, self.decv_leaky_ratio[layer_id], self.output_low_bound, self.output_up_bound) # Nonlinear act
            else:
                #net = ne.leaky_brelu(net, self.decv_leaky_ratio[layer_id], self.output_low_bound, self.output_up_bound) # Nonlinear act
                net = ne.leaky_relu(net, self.decv_leaky_ratio[layer_id])
                #net = ne.elu(net)

        net = tf.identity(net, name='output')
        net = tf.reshape(net, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        #import pdb; pdb.set_trace()
        return net

    
    @lazy_method
    def encoder(self, inputs):
        conv = self.conv_layers(inputs)
        assert conv.get_shape().as_list()[1:] == self.conv_out_shape
        enfc = self.enfc_layers(conv)
        assert enfc.get_shape().as_list()[1:] == [self.center_state_size]
        return enfc
    

    @lazy_method
    def decoder(self, inputs):
        defc = self.defc_layers(inputs)
        assert defc.get_shape().as_list()[1:] == self.decv_in_shape
        generated = self.decv_layers(defc)
        assert generated.get_shape().as_list()[1:] == [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS]
        return generated


    @lazy_method
    def prediction(self, data, is_training):
        self.is_training = is_training
        states = self.encoder(data)
        generated = self.decoder(states)
        return generated

    def tf_load(self, sess, path, name='deep_cae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, name='deep_cae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.save(sess, path+'/'+name+spec)
