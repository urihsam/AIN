""" Channel-attentive Variational Convolutional Autoconvder
"""
import vcae
from dependency import *
import utils.net_element as ne
from utils.decorator import lazy_method, lazy_property, lazy_method_no_scope


class CAVCAE:
    def __init__(self, 
                 # relu bounds
                 output_low_bound, 
                 output_up_bound,
                 # conv layers
                 conv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[64, 64, 64, 3],
                 conv_leaky_ratio=[0.2, 0.2, 0.2, 0.1],
                 # deconv layers
                 decv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 decv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 decv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 decv_channel_sizes=[3, 64, 64, 64],  # [1, 128, 128, 128, 128]
                 decv_leaky_ratio=[0.1, 0.2, 0.2, 0.01],
                 # encoder fc layers
                 enfc_state_sizes=[4096], 
                 enfc_leaky_ratio=[0.2, 0.2],
                 enfc_drop_rate=[0, 0.75],
                 # bottleneck
                 central_state_size=2048, 
                 # decoder fc layers
                 defc_state_sizes=[4096],
                 defc_leaky_ratio=[0.2, 0.2],
                 defc_drop_rate=[0.75, 0],
                 # img channel
                 img_channel=None,
                 # switch
                 use_batch_norm = False
                ):
        
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        if self.img_channel == 1:
            with tf.variable_scope("cavcae0"):
                self._model = vcae.VCAE(output_low_bound, output_up_bound, 
                conv_filter_sizes, conv_strides, conv_padding, conv_channel_sizes, conv_leaky_ratio,
                decv_filter_sizes, decv_strides, decv_padding, decv_channel_sizes, decv_leaky_ratio,
                enfc_state_sizes, enfc_leaky_ratio, enfc_drop_rate, central_state_size, 
                defc_state_sizes, defc_leaky_ratio, defc_drop_rate, 
                1, use_batch_norm)
        else:
            self._model = []
            for m_idx in range(self.img_channel):
                with tf.variable_scope("cavcae{}".format(m_idx)):
                    m = vcae.VCAE(output_low_bound, output_up_bound, 
                            conv_filter_sizes, conv_strides, conv_padding, conv_channel_sizes, conv_leaky_ratio,
                            decv_filter_sizes, decv_strides, decv_padding, decv_channel_sizes, decv_leaky_ratio,
                            enfc_state_sizes, enfc_leaky_ratio, enfc_drop_rate, central_state_size, 
                            defc_state_sizes, defc_leaky_ratio, defc_drop_rate, 
                            1, use_batch_norm)
                    self._model.append(m)
        

    
    @lazy_method
    def prediction(self, data, is_training):
        if self.img_channel == 1:
            attentive = self._model.prediction(data, is_training)
        else:
            data_splits = tf.split(data, self.img_channel, 3)
            gen_splits = []; sum_splits = []
            for m_idx in range(self.img_channel):
                gen = tf.squeeze(self._model[m_idx].prediction(data_splits[m_idx], is_training), 3)
                gen_splits.append(gen)
                sum_splits.append(tf.reduce_sum(gen))
            generated = tf.stack(gen_splits, 3)
            #sums = tf.stack(sum_splits)
            #self.attention = sums / tf.reduce_sum(sums)
            #attentive = tf.multiply(generated, self.attention)
            attentive = generated

        return attentive


    @lazy_method
    def kl_distance(self):
        dists = []
        for m_idx in range(self.img_channel):
            dists.append(self._model[m_idx].kl_distance())
        #loss = tf.multiply(tf.stack(dists), self.attention)
        loss = tf.stack(dists)
        return tf.reduce_mean(loss)
    

    def tf_load(self, sess, path, name='deep_cavcae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, name='deep_cavcae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.save(sess, path+'/'+name+spec)