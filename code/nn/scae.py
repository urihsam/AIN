""" Sparse Convolutional Autoconvder
"""
from nn.cae import CAE
from dependency import *
import utils.net_element as ne
from utils.decorator import lazy_method, lazy_property, lazy_method_no_scope


class SCAE(CAE):
    def __init__(self, 
                 output_low_bound, 
                 output_up_bound,
                 # relu bounds
                 nonlinear_low_bound,
                 nonlinear_up_bound,
                 # conv layers
                 conv_filter_size=[3,3], 
                 conv_channel_sizes=[128, 128, 128, 128, 1], #[256, 256, 256, 1]
                 conv_leaky_ratio=[0.2, 0.4, 0.4, 0.2, 0.2],
                 # deconv layers
                 decv_filter_size=[3,3], 
                 decv_channel_sizes=[1, 128, 128, 128, 128], #[1, 256, 256, 256]
                 decv_leaky_ratio=[0.2, 0.2, 0.4, 0.4, 0],
                 # encoder fc layers
                 enfc_state_sizes=[4096], 
                 enfc_leaky_ratio=[0.2, 0.2],
                 enfc_drop_rate=[0, 0.75],
                 # bottleneck
                 central_state_size=1024, 
                 # decoder fc layers
                 defc_state_sizes=[4096],
                 defc_leaky_ratio=[0.2, 0.2],
                 defc_drop_rate=[0.75, 0],
                 # switch
                 use_batch_norm = False
                ):
        
        super().__init__(output_low_bound, output_up_bound, nonlinear_low_bound, nonlinear_up_bound,
              conv_filter_size, conv_channel_sizes, conv_leaky_ratio,
              decv_filter_size, decv_channel_sizes, decv_leaky_ratio,
              enfc_state_sizes, enfc_leaky_ratio, enfc_drop_rate, central_state_size, 
              defc_state_sizes, defc_leaky_ratio, defc_drop_rate, use_batch_norm
             )

    
    @lazy_method
    def prediction(self, data, is_training):
        self.is_training = is_training
        self.states = self.encoder(data)
        generated = self.decoder(self.states)
        return generated
    

    @lazy_method
    def rho_distance(self, true_rho):
        return tf.maximum(tf.reduce_mean(self.states), true_rho)

