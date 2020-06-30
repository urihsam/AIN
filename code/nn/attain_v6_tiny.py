from utils.fgsm_attack import fgm
import nn.cae as cae
import nn.scae as scae
import nn.vcae as vcae
import nn.vcae_new as vcae_new
import nn.cavcae as cavcae
import nn.resnet as resnet
import nn.ae.attresenc_v4 as attresenc
#import nn.ae.attresdec_v4 as attresdec
import nn.ae.attresdec_v5 as attresdec
import nn.embedder as embedder
from utils.decorator import *
from dependency import *
import os, math


class ATTAIN:
    """
    Attentive Adversarial Imitation Network, targeted/untargeted
    """

    def __init__(self, data, label, low_bound, up_bound, attack_epsilon, is_training, data_fake=None, targeted_label=None):
        self.data = data
        self.label = label
        self.output_low_bound = low_bound
        self.output_up_bound = up_bound
        self.central_emb_size = FLAGS.EMB_SIZE
        self.central_channel_size = FLAGS.CENTRAL_CHANNEL_SIZE
        self.attack_epsilon = attack_epsilon
        self.is_training = is_training
        self.targeted_label = targeted_label

        # Class label autoencoder
        with tf.variable_scope(FLAGS.LBL_NAME) as lbl_scope:
            if FLAGS.LABEL_CONDITIONING:
                with tf.variable_scope("conditional"):
                    row = self.data.get_shape().as_list()[1]#data.get_shape().as_list()[1]
                    col = self.data.get_shape().as_list()[2]#int(math.ceil(FLAGS.NUM_CLASSES/data.get_shape().as_list()[1]))
                    cha = 1#self.data.get_shape().as_list()[3]
                    img_shape = [row, col, cha]
                    size = np.prod(img_shape)
                    w1_shape = [label.get_shape().as_list()[-1], size]
                    w1 = ne.weight_variable(w1_shape, "lbl_W1", init_type="HE")
                    b1 = ne.bias_variable([size], "lbl_b1")
                    label_states = ne.sigmoid(ne.fully_conn(label, w1, b1))
                    label_states = ne.layer_norm(label_states, self.is_training)
                    #
                    w2_shape = [size, label.get_shape().as_list()[-1]]
                    w2 = ne.weight_variable(w2_shape, "lbl_W2", init_type="HE")
                    b2 = ne.bias_variable([label.get_shape().as_list()[-1]], "lbl_b2")
                    self._recon_label = ne.sigmoid(ne.fully_conn(label_states, w2, b2))
                    #
                    self._label_states = tf.reshape(label_states, [-1] + img_shape)
                    #self.labeled_data = tf.multiply(self.data, self._label_states) # 2
                    # self.labeled_data = tf.concat([self.data, self._label_states], -1) # 1

                if FLAGS.USE_LABEL_MASK:
                    with tf.variable_scope("mask"):
                        ## mask
                        row = 64
                        col = 64
                        img_shape = [row, col]
                        size = np.prod(img_shape)
                        w1_shape = [label.get_shape().as_list()[-1], size]
                        w1 = ne.weight_variable(w1_shape, "mask_W1", init_type="HE")
                        b1 = ne.bias_variable([size], "mask_b1")
                        label_states = ne.tanh(ne.fully_conn(label, w1, b1))
                        label_states = ne.layer_norm(label_states, self.is_training)
                        #
                        w2_shape = [size, label.get_shape().as_list()[-1]]
                        w2 = ne.weight_variable(w2_shape, "mask_W2", init_type="HE")
                        b2 = ne.bias_variable([label.get_shape().as_list()[-1]], "mask_b2")
                        self._recon_label_2 = ne.sigmoid(ne.fully_conn(label_states, w2, b2))
                        #
                        self._mask_states = tf.reshape(label_states, [-1] + img_shape)
                else:
                    self._mask_states = None
            else:
                self._label_states = tf.reshape(label_states, [-1] + img_shape)
                self._mask_states = None
            
            if FLAGS.ADD_RANDOM:
                with tf.variable_scope("random"):
                    row = 8
                    col = 8
                    cha = 1024
                    img_shape = [row, col, cha]
                    size = np.prod(img_shape)
                    w1_shape = [label.get_shape().as_list()[-1], size]
                    w1 = ne.weight_variable(w1_shape, "lbl_W1", init_type="HE")
                    b1 = ne.bias_variable([size], "lbl_b1")
                    label_states = ne.sigmoid(ne.fully_conn(label, w1, b1))
                    label_states = ne.layer_norm(label_states, self.is_training)
                    #
                    w2_shape = [size, label.get_shape().as_list()[-1]]
                    w2 = ne.weight_variable(w2_shape, "lbl_W2", init_type="HE")
                    b2 = ne.bias_variable([label.get_shape().as_list()[-1]], "lbl_b2")
                    self._recon_label_3 = ne.sigmoid(ne.fully_conn(label_states, w2, b2))
                    #
                    self._random_states = tf.reshape(label_states, [-1] + img_shape)
            else:
                self._random_states = None

        with tf.variable_scope('autoencoder'):
            if FLAGS.AE_TYPE == "ATTAE":
                # Encoder
                with tf.variable_scope(FLAGS.ENC_NAME) as enc_scope:
                    self._encoder = attresenc.ATTRESENC(
                        attention_type=FLAGS.ATT_TYPE,
                        att_pos_idx=1,
                        att_f_channel_size = 64,
                        att_g_channel_size = 64,
                        att_h_channel_size = 64,
                        conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
                        conv_strides = [[2,2], [1,1], [2,2], [1,1], [2,2]], 
                        conv_channel_sizes=[64, 128, 256, 512, 1024], 
                        conv_leaky_ratio=[0.2, 0.2, 0.2, 0.2, 0.2],
                        conv_drop_rate=[0.0, 0.8, 0.2, 0.6, 0.0], #1.[0.0, 0.8, 0.2, 0.6, 0.0], 2.[0.0, 0.8, 0.2, 0.6, 0.0]
                        num_res_block=FLAGS.NUM_ENC_RES_BLOCK,
                        res_block_size=FLAGS.ENC_RES_BLOCK_SIZE,
                        res_filter_sizes=[1,1],
                        res_leaky_ratio=0.2,
                        res_drop_rate=0.4,
                        out_channel_size=self.central_channel_size,
                        # random noise
                        add_random_noise=FLAGS.ADD_RANDOM, mean=0.0, stddev=0.01,
                        out_norm=FLAGS.ENC_OUT_NORM,
                        use_norm=FLAGS.ENC_NORM,
                        img_channel=FLAGS.NUM_CHANNELS)
                    self._central_states = self._encoder.evaluate(self.data, self.is_training, 
                                           self._label_states, self._random_states)
                
                # Decoder
                with tf.variable_scope(FLAGS.DEC_NAME) as dec_scope:
                    self._decoder_t = attresdec.ATTRESDEC(
                        self.output_low_bound, self.output_up_bound,
                        attention_type=FLAGS.ATT_TYPE,
                        att_pos_idx=-2,
                        att_f_channel_size = 64,
                        att_g_channel_size = 64,
                        att_h_channel_size = 64,
                        decv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
                        decv_strides = [[2,2], [1,1], [2,2], [1,1], [2,2]], 
                        decv_channel_sizes=[1024, 512, 256, 128, 64], 
                        decv_leaky_ratio=[0.2, 0.2, 0.2, 0.2, 0.2],
                        decv_drop_rate=[0.0, 0.6, 0.2, 0.8, 0.0], #1.[0.0, 0.6, 0.2, 0.8, 0.0], 2.[0.0, 0.6, 0.2, 0.8, 0.0]
                        num_res_block=FLAGS.NUM_DEC_RES_BLOCK,
                        res_block_size=FLAGS.DEC_RES_BLOCK_SIZE,
                        res_filter_sizes=[1,1],
                        res_leaky_ratio=0.2,
                        res_drop_rate=0.4,
                        in_channel_size=self.central_channel_size,
                        in_norm=FLAGS.DEC_IN_NORM,
                        use_norm=FLAGS.DEC_NORM)
                    self._generated_t = self._decoder_t.evaluate(self._central_states, self.is_training, self._mask_states)
                    #self._generated_t = tf.reshape(self._generated_t, [-1, row, data.get_shape().as_list()[2]+col, data.get_shape().as_list()[3]])
                    #self._generated_t, self._generated_label_states = tf.split(self._generated_t, [data.get_shape().as_list()[2], col], -2)                
                generated = self._generated_t + data
                
                if FLAGS.NORMALIZE:
                    self._autoencoder_prediction = ne.brelu(generated)
                else:
                    self._autoencoder_prediction = ne.brelu(generated, low_bound=0, up_bound=255)
                
        
        # Adv data generated by AE
        with tf.variable_scope('target') as scope:
            self._target_adv = resnet.resnet18()
            self._target_adv_logits, self._target_adv_prediction = self._target_adv.prediction(self._autoencoder_prediction)
            if FLAGS.IS_TARGETED_ATTACK or self.targeted_label != None: # targeted
                self._target_adv_accuracy = self._target_adv.accuracy(self._target_adv_prediction, self.targeted_label)
            else:
                self._target_adv_accuracy = self._target_adv.accuracy(self._target_adv_prediction, self.label)
                
        # Fake data generated by attacking algo
        self.target_scope = scope
        with tf.variable_scope(scope, reuse=True):
            self._target_attack = self._target_adv
            if data_fake == None:
                if FLAGS.IS_TARGETED_ATTACK or self.targeted_label != None: # targeted
                    self.data_fake = fgm(self._target_attack.prediction, data, self.targeted_label, eps=self.attack_epsilon, iters=FLAGS.FGM_ITERS,
                                        clip_min=0., clip_max=1.)
                else:
                    self.data_fake = fgm(self._target_attack.prediction, data, label, eps=self.attack_epsilon, iters=FLAGS.FGM_ITERS,
                                        targeted=False, clip_min=0., clip_max=1.)
            else:
                self.data_fake = data_fake
        
        with tf.variable_scope(scope, reuse=True):
            self._target_fake = self._target_adv
            self._target_fake_logits, self._target_fake_prediction = self._target_fake.prediction(self.data_fake)
            self.label_fake = self.get_label(self._target_fake_prediction)
            if FLAGS.IS_TARGETED_ATTACK or self.targeted_label != None: # targeted
                self._target_fake_accuracy = self._target_fake.accuracy(self._target_fake_prediction, self.targeted_label)
            else:
                self._target_fake_accuracy = self._target_fake.accuracy(self._target_fake_prediction, label)
                
        # Clean data
        with tf.variable_scope(scope, reuse=True):
            self._target = self._target_adv
            self._target_logits, self._target_prediction = self._target.prediction(data)
            self._target_accuracy = self._target.accuracy(self._target_prediction, label)
            
        self.prediction


    def get_label(self, prediction):
        n_class = prediction.get_shape().as_list()[1]
        indices = tf.argmax(prediction, axis=1)
        return tf.one_hot(indices, n_class, on_value=1.0, off_value=0.0)


    def vectorize(self, x):
        return tf.reshape(x, [-1, FLAGS.IMAGE_ROWS * FLAGS.IMAGE_COLS * FLAGS.NUM_CHANNELS])
        

    @lazy_method
    def loss_x(self, beta_t, beta_f):
        x_adv = self.vectorize(self._autoencoder_prediction) # adv
        x_true = self.vectorize(self.data)
        x_fake = self.vectorize(self.data_fake)
        
        max_dist_true = tf.reduce_max(tf.abs(x_adv-x_true))
        max_dist_fake = tf.reduce_max(tf.abs(x_adv-x_fake))
        # print info
        max_dist_trans = tf.reduce_max(tf.abs(x_true-x_fake))
        def _dist(vec1, vec2):
            if FLAGS.NORM_TYPE == "L2":
                Lx_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(vec1-vec2), 1)))
            if FLAGS.NORM_TYPE == "L1":
                Lx_dist = tf.reduce_mean(tf.reduce_sum(tf.abs(vec1-vec2), 1))
            elif FLAGS.NORM_TYPE == "INF":
                Lx_dist = tf.reduce_mean(tf.reduce_max(tf.abs(vec1-vec2), 1))
            return Lx_dist
            
        Lx_dist_true = _dist(x_adv, x_true)
        Lx_true = beta_t * Lx_dist_true

        Lx_dist_fake = _dist(x_adv, x_fake)
        if FLAGS.USE_IMITATION:
            Lx_fake = beta_f * Lx_dist_fake
            if FLAGS.ONLY_IMITATION:
                Lx_true = tf.constant(0.0)
        else:
            Lx_fake = tf.constant(0.0)

        Lx = Lx_true + Lx_fake

        # print info
        Lx_dist_trans = _dist(x_true, x_fake)

        #tf.summary..scalar("Loss_x", Lx)
        #tf.summary..scalar("Loss_x_true", Lx_true)
        #tf.summary..scalar("Loss_x_fake", Lx_fake)
        #tf.summary..scalar("Dist_x_true", Lx_dist_true)
        #tf.summary..scalar("Dist_x_fake", Lx_dist_fake)
        #tf.summary..scalar("Dist_x_trans", Lx_dist_trans)
        #tf.summary..scalar("Max_pixel_dist_true", max_dist_true)
        #tf.summary..scalar("Max_pixel_dist_fake", max_dist_fake)
        #tf.summary..scalar("Max_pixel_dist_trans", max_dist_trans)
        return Lx, (Lx_true, Lx_fake), (Lx_dist_true, Lx_dist_fake, Lx_dist_trans), (max_dist_true, max_dist_fake, max_dist_trans)


    @lazy_property
    def max_distance(self):
        x_adv = self.vectorize(self._autoencoder_prediction) # adv
        x_true = self.vectorize(self.data)
        x_fake = self.vectorize(self.data_fake)
        
        max_dist_true = tf.reduce_max(tf.abs(x_adv-x_true))
        max_dist_fake = tf.reduce_max(tf.abs(x_adv-x_fake))

        #tf.summary..scalar("Max_dist_true", max_dist_true)
        #tf.summary..scalar("Max_dist_fake", max_dist_fake)
        return max_dist_true, max_dist_fake


    @lazy_property
    def norm_distance(self):
        def norm_dist(vec1, vec2):
            if FLAGS.NORM_TYPE == "L2":
                Lx_dist = tf.reduce_mean(tf.norm(vec1-vec2, ord=2, axis=1))
            if FLAGS.NORM_TYPE == "L1":
                Lx_dist = tf.reduce_mean(tf.norm(vec1-vec2, ord=1, axis=1))
            elif FLAGS.NORM_TYPE == "INF":
                Lx_dist = tf.reduce_mean(tf.norm(vec1-vec2, ord=np.inf, axis=1))
            return Lx_dist
        x_adv = self.vectorize(self._autoencoder_prediction) # adv
        x_true = self.vectorize(self.data)
        x_fake = self.vectorize(self.data_fake)

        norm_dist_adv_true = norm_dist(x_adv, x_true)
        norm_dist_adv_fake = norm_dist(x_adv, x_fake)

        #tf.summary..scalar("Norm_dist_adv_true", norm_dist_adv_true)
        #tf.summary..scalar("Norm_dist_adv_fake", norm_dist_adv_fake)
        return norm_dist_adv_true, norm_dist_adv_fake
    

    @lazy_method
    def loss_y(self, beta_t, beta_f, beta_f2, beta_c):
        def loss_y_from_trans(): 
            if FLAGS.IS_TARGETED_ATTACK or self.targeted_label != None: # targeted
                y_faked = tf.argmax(self.targeted_label, axis=1, output_type=tf.int32)
            else:
                y_faked = tf.argmax(self._target_fake_logits, axis=1, output_type=tf.int32)
            mask1 = tf.one_hot(y_faked, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
            y_clean = tf.argmax(self.label, axis=1, output_type=tf.int32)
            mask2 = tf.one_hot(y_clean, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
            adv_logits_at_y_faked = tf.reduce_max(tf.subtract(self._target_adv_logits, mask1), axis=1)
            adv_logits_at_y_clean = tf.reduce_max(tf.subtract(self._target_adv_logits, mask2), axis=1)
            if FLAGS.LOSS_MODE_TRANS == "C_W": # Make the logits at the largest prob position of fake data larger than that at true position
                Ly_dist = tf.reduce_mean(
                    tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_at_y_faked), FLAGS.LOSS_Y_LOW_BOUND_T)
                )
            elif FLAGS.LOSS_MODE_TRANS == "C_W2": # Make the logits at the largest prob position of fake data larger than that at true position
                loss_1 = tf.reduce_mean(tf.maximum(
                    tf.minimum(tf.subtract(adv_logits_at_y_clean, adv_logits_at_y_faked), 0.0),
                    FLAGS.LOSS_Y_LOW_BOUND_T)
                )
                loss_2 = tf.reduce_mean(tf.minimum(FLAGS.LOSS_Y_UP_BOUND_T,
                        np.exp(1.0) * (
                            tf.exp(
                                tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_at_y_faked), 0.0)
                            ) - 1.0)
                    )
                )
                Ly_dist = loss_1 + loss_2
            elif FLAGS.LOSS_MODE_TRANS == "C_W3": # Make the logits at the largest prob position of fake data larger than that at true position
                loss_1 = tf.reduce_mean(tf.maximum(
                    tf.minimum(tf.subtract(adv_logits_at_y_clean, adv_logits_at_y_faked), 0.0),
                    FLAGS.LOSS_Y_LOW_BOUND_T)
                )
                loss_2 = tf.reduce_mean(tf.minimum(FLAGS.LOSS_Y_UP_BOUND_T,
                        np.exp(1.0) * (
                            tf.exp(
                                tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_at_y_faked), 0.0)
                            ) - 1.0)
                    )
                )
                Ly_dist = loss_2
                
            return beta_t * Ly_dist, Ly_dist


        def loss_y_from_fake(): 
            loss_logits =tf.constant(0.0)
            if FLAGS.IS_TARGETED_ATTACK or self.targeted_label != None: # targeted
                y_faked = tf.argmax(self.targeted_label, axis=1, output_type=tf.int32)
            else:
                y_faked = tf.argmax(self._target_fake_logits, axis=1, output_type=tf.int32)
            mask1 = tf.one_hot(y_faked, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
            mask2 = tf.one_hot(y_faked, FLAGS.NUM_CLASSES, on_value=float('inf'), off_value=0.0)
            adv_logits_at_y_faked = tf.reduce_max(tf.subtract(self._target_adv_logits, mask1), axis=1)
            adv_logits_max_exclude_y_faked = tf.reduce_max(tf.subtract(self._target_adv_logits, mask2), axis=1)
            if FLAGS.LOSS_MODE_FAKE == "C_W": # Maximize the logits at max prob position of faked data
                Ly_dist = tf.reduce_mean(
                    tf.maximum(tf.subtract(adv_logits_max_exclude_y_faked, adv_logits_at_y_faked), FLAGS.LOSS_Y_LOW_BOUND_F)
                )
            elif FLAGS.LOSS_MODE_FAKE == "C_W2": # Maximize the logits at max prob position of faked data
                loss_1 = tf.reduce_mean(tf.maximum(
                    tf.minimum(tf.subtract(adv_logits_max_exclude_y_faked, adv_logits_at_y_faked), 0.0),
                    FLAGS.LOSS_Y_LOW_BOUND_T)
                )
                loss_2 = tf.reduce_mean(tf.minimum(FLAGS.LOSS_Y_UP_BOUND_T,
                        np.exp(1.0) * (
                            tf.exp(
                                tf.maximum(tf.subtract(adv_logits_max_exclude_y_faked, adv_logits_at_y_faked), 0.0)
                            ) - 1.0)
                    )
                )
                Ly_dist = loss_1 + loss_2
            elif FLAGS.LOSS_MODE_FAKE == "MIX":
                loss_1 = tf.reduce_mean(tf.maximum(
                    tf.minimum(tf.subtract(adv_logits_max_exclude_y_faked, adv_logits_at_y_faked), 0.0),
                    FLAGS.LOSS_Y_LOW_BOUND_T)
                )
                loss_2 = tf.reduce_mean(tf.minimum(FLAGS.LOSS_Y_UP_BOUND_T,
                        np.exp(1.0) * (
                            tf.exp(
                                tf.maximum(tf.subtract(adv_logits_max_exclude_y_faked, adv_logits_at_y_faked), 0.0)
                            ) - 1.0)
                    )
                )
                Ly_dist = loss_1 + loss_2

                
                '''
                if self.targeted_label is not None: # targeted
                #if self.targeted_label is  None:
                    cond = tf.expand_dims(
                        tf.cast(tf.equal(
                            tf.argmax(self._target_fake_logits, axis=1, output_type=tf.int32),
                            tf.argmax(self.targeted_label, axis=1, output_type=tf.int32)
                        ), tf.float32),
                        axis=1)
                else: # untargeted
                    cond = tf.expand_dims(
                        tf.cast(tf.not_equal(
                            tf.argmax(self._target_fake_logits, axis=1, output_type=tf.int32),
                            tf.argmax(self.label, axis=1, output_type=tf.int32)
                        ), tf.float32),
                        axis=1)
                loss_logits = tf.reduce_mean(
                    tf.multiply(cond, 
                        tf.sqrt(
                            tf.reduce_sum(tf.square(
                                self._target_adv_logits-self._target_fake_logits
                                ), 1)
                        )
                    )
                )
                '''
                # no matter targeted or untargeted, we don't want predicted label to be equal to label
                cond = tf.expand_dims(
                    tf.cast(tf.not_equal(
                        tf.argmax(self._target_fake_logits, axis=1, output_type=tf.int32),
                        tf.argmax(self.label, axis=1, output_type=tf.int32)
                    ), tf.float32),
                    axis=1)
                loss_logits = tf.reduce_mean(
                    tf.multiply(cond, 
                        tf.sqrt(
                            tf.reduce_sum(tf.square(
                                self._target_adv_logits-self._target_fake_logits
                                ), 1)
                        )
                    )
                )
                
            if FLAGS.USE_IMITATION:
                return beta_f * Ly_dist + beta_f2 * loss_logits, Ly_dist
            else:
                return beta_f * Ly_dist, Ly_dist
        
        
        def loss_y_from_clean(): 
            y_clean = tf.argmax(self.label, axis=1, output_type=tf.int32)
            mask1 = tf.one_hot(y_clean, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
            mask2 = tf.one_hot(y_clean, FLAGS.NUM_CLASSES, on_value=float('inf'), off_value=0.0)
            adv_logits_at_y_clean = tf.reduce_max(tf.subtract(self._target_adv_logits, mask1), axis=1)
            adv_logits_max_exclude_y_clean = tf.reduce_max(tf.subtract(self._target_adv_logits, mask2), axis=1)
            if FLAGS.LOSS_MODE_CLEAN == "C_W": # Minimize the logits at clean label
                Ly_dist = tf.reduce_mean(
                    tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_max_exclude_y_clean), FLAGS.LOSS_Y_LOW_BOUND_C)
                )
            
            elif FLAGS.LOSS_MODE_CLEAN == "C_W2": # Minimize the logits at clean label
                # part 1
                loss_1 = tf.reduce_mean(tf.maximum(
                    tf.minimum(tf.subtract(adv_logits_at_y_clean, adv_logits_max_exclude_y_clean), 0.0),
                    FLAGS.LOSS_Y_LOW_BOUND_C)
                )

                loss_2 = tf.reduce_mean(tf.minimum(FLAGS.LOSS_Y_UP_BOUND_C,
                        np.exp(1.0) * (
                            tf.exp(
                                tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_max_exclude_y_clean), 0.0)
                            ) -1.0
                        )
                    )
                )
                Ly_dist = loss_1 + loss_2
            
            elif FLAGS.LOSS_MODE_CLEAN == "C_W3": # Minimize the logits at clean label
                # part 1
                loss_1 = tf.reduce_mean(tf.maximum(
                    tf.minimum(tf.subtract(adv_logits_at_y_clean, adv_logits_max_exclude_y_clean), 0.0),
                    FLAGS.LOSS_Y_LOW_BOUND_C)
                )

                loss_2 = tf.reduce_mean(tf.minimum(FLAGS.LOSS_Y_UP_BOUND_C,
                        np.exp(1.0) * (
                            tf.exp(
                                tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_max_exclude_y_clean), 0.0)
                            ) -1.0
                        )
                    )
                )
                Ly_dist = loss_2
                
                
            return beta_c * Ly_dist, Ly_dist

        Ly_trans, Ly_dist_trans = loss_y_from_trans()
        Ly_fake, Ly_dist_fake = loss_y_from_fake()
        Ly_clean, Ly_dist_clean = loss_y_from_clean()
        if FLAGS.IS_TARGETED_ATTACK == False and FLAGS.USE_IMITATION == False: # untargeted and no imitation
            Ly_fake = tf.constant(0.0)
            Ly_trans = tf.constant(0.0)
        if FLAGS.USE_IMITATION and FLAGS.ONLY_IMITATION:
            Ly_trans = tf.constant(0.0)
            Ly_clean = tf.constant(0.0)
        
        Ly = Ly_trans + Ly_fake + Ly_clean

        #tf.summary..scalar("Loss_y_trans", Ly_trans)
        #tf.summary..scalar("Dist_y_trans", Ly_dist_trans)
        #tf.summary..scalar("Loss_y_fake", Ly_fake)
        #tf.summary..scalar("Dist_y_fake", Ly_dist_fake)
        #tf.summary..scalar("Loss_y_clean", Ly_clean)
        #tf.summary..scalar("Dist_y_clean", Ly_dist_clean)
        return Ly, (Ly_trans, Ly_fake, Ly_clean), (Ly_dist_trans, Ly_dist_fake, Ly_dist_clean)

    
    @lazy_property
    def loss_reconstruct(self):
        adv = self.vectorize(self._autoencoder_prediction)
        fake = self.vectorize(self.data_fake)
        self.adv = adv
        self.fake = fake
        cross_entropy = -tf.reduce_sum(
            fake * tf.log(1e-5+adv) + (1-fake) * tf.log(1e-5+1-adv), 
            axis=1
        )
        self.cross_entropy = cross_entropy
        return tf.reduce_mean(cross_entropy)


    @lazy_property
    def pre_loss_label(self):
        loss = 0
        if FLAGS.LABEL_CONDITIONING:
            cross_entropy = -tf.reduce_sum(
                self.label * tf.log(1e-5+self._recon_label) + (1-self.label) * tf.log(1e-5 + 1-self._recon_label),
                axis = 1
            )
            loss += tf.reduce_mean(cross_entropy)
            if FLAGS.USE_LABEL_MASK:
                cross_entropy_2 = -tf.reduce_sum(
                    self.label * tf.log(1e-5+self._recon_label_2) + (1-self.label) * tf.log(1e-5 + 1-self._recon_label_2),
                    axis = 1
                )
                loss += tf.reduce_mean(cross_entropy_2)
        if FLAGS.ADD_RANDOM:
            cross_entropy_3 = -tf.reduce_sum(
                self.label * tf.log(1e-5+self._recon_label_3) + (1-self.label) * tf.log(1e-5 + 1-self._recon_label_3),
                axis = 1
            )
            loss += tf.reduce_mean(cross_entropy_3)
        return loss


    
    @lazy_property
    def loss_label_states(self):
        squared_sum = tf.reduce_sum(
            tf.square(self._label_states - self._generated_label_states),
            axis = 1
        )
        return tf.reduce_mean(squared_sum)
    

    @lazy_method
    def loss(self, partial_loss, loss_x, loss_y):
        partial_loss_func = lambda: tf.cond(tf.equal(partial_loss, "LOSS_X"), lambda: loss_x, lambda: loss_y)
        loss = tf.cond(tf.equal(partial_loss, "FULL_LOSS"), lambda: loss_x + loss_y, partial_loss_func)
        recon_loss = FLAGS.GAMMA_R * self.loss_reconstruct
        label_loss = tf.constant(0.0) #FLAGS.GAMMA_L * self.loss_label_states
        loss += (recon_loss + label_loss)
        if FLAGS.SPARSE:
            sparse_loss = FLAGS.GAMMA_S * self._autoencoder.rho_distance(FLAGS.SPARSE_RHO)
            loss += sparse_loss
            variational_loss = tf.constant(0)
        elif FLAGS.VARI:
            variational_loss = FLAGS.GAMMA_V * self._autoencoder.kl_distance()
            loss += variational_loss
            sparse_loss = tf.constant(0)
        else:
            sparse_loss = tf.constant(0)
            variational_loss = tf.constant(0)

        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder")
        if FLAGS.REG_SCALE is not None:
            regularize = tf.contrib.layers.l1_l2_regularizer(FLAGS.REG_SCALE, FLAGS.REG_SCALE)
            #print tf.GraphKeys.TRAINABLE_VARIABLES
            reg_term = sum([regularize(param) for param in opt_vars])
            loss += reg_term
            #tf.summary..scalar("Regularization", reg_term)
        #tf.summary..scalar("Total_loss", loss)
        #tf.summary..scalar("Reconstruct_loss", recon_loss)
        #tf.summary..scalar("Label_loss", label_loss)
        #tf.summary..scalar("Sparse_loss", sparse_loss)
        #tf.summary..scalar("Variational_loss", variational_loss)
        return loss, recon_loss, label_loss, sparse_loss, variational_loss, reg_term

    @lazy_method_no_scope
    def compute_gradients(self, loss, scope="CMP_GRADS"):
        with tf.variable_scope(scope):
            """
            decayed_learning_rate = learning_rate *
                            decay_rate ^ (global_step / decay_steps)
            learning rate decay with decay_rate per decay_steps
            """
            # decay
            learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE, global_step, 
                FLAGS.LEARNING_DECAY_STEPS, FLAGS.LEARNING_DECAY_RATE)

            if scope == "PRE_OPT":
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, FLAGS.LBL_NAME)
            else:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder")
                lbl_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, FLAGS.LBL_NAME)
                opt_vars = list(set(opt_vars) - set(lbl_vars))

            momentum = 0.9
            """optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(
                loss, var_list=opt_vars)"""
            if FLAGS.OPT_TYPE == "NEST":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
            elif FLAGS.OPT_TYPE == "MOME":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=False)
            elif FLAGS.OPT_TYPE == "ADAM":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss, var_list=opt_vars)

            return grads_and_vars, optimizer, learning_rate

        
    @lazy_method_no_scope
    def apply_gradients(self, grads_and_vars, optimizer, accum_iters=1, scope="APY_GRADS"):
        with tf.variable_scope(scope):
            # use lobal step to keep track of our iterations
            global_step = tf.Variable(0, name="OPT_GLOBAL_STEP", trainable=False)

            gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
            # accumulate
            if accum_iters != 1:
                accum_grads = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in variables]                                        
                
                zero_op = [grads.assign(tf.zeros_like(grads)) for grads in accum_grads]
                accum_op = [accum_grads[i].assign_add(g) for i, g in enumerate(gradients) if g!= None]
                avg_op = [grads.assign(grads/accum_iters) for grads in accum_grads]
            else:
                zero_op = None
                accum_op = None
                avg_op = None
                accum_grads = gradients
            if FLAGS.IS_GRAD_CLIPPING:
                """
                https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
                clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
                t_list[i] * clip_norm / max(global_norm, clip_norm),
                where global_norm = sqrt(sum([l2norm(t)**2 for t in t_list])
                if clip_norm < global_norm, the gradients will be scaled to smaller values,
                especially, if clip_norm == 1, the graidents will be normed
                """
                clipped_grads, global_norm = (
                    #tf.clip_by_global_norm(gradients) )
                    tf.clip_by_global_norm(accum_grads, clip_norm=FLAGS.GRAD_CLIPPING_NORM))
                grads_and_vars = zip(clipped_grads, variables)
            else:
                grads_and_vars = zip(accum_grads, variables)
            op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # tensorboard
            for g, v in grads_and_vars:
                # print v.name
                name = v.name.replace(":", "_")
                #tf.summary..histogram(name+"_gradients", g)
            #tf.summary..scalar("Learning_rate", learning_rate)
            return op, (zero_op, accum_op, avg_op)
        

    @lazy_method_no_scope
    def optimization(self, loss, accum_iters=1, scope="OPT"):
        with tf.variable_scope(scope):
            """
            decayed_learning_rate = learning_rate *
                            decay_rate ^ (global_step / decay_steps)
            learning rate decay with decay_rate per decay_steps
            """
            # use lobal step to keep track of our iterations
            global_step = tf.Variable(0, name="OPT_GLOBAL_STEP", trainable=False)

            # reset global step
            reset_decay_op = global_step.assign(tf.constant(0))
            # decay
            learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE, global_step, 
                FLAGS.LEARNING_DECAY_STEPS, FLAGS.LEARNING_DECAY_RATE)

            if scope == "PRE_OPT":
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, FLAGS.LBL_NAME)
            else:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder")
                lbl_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, FLAGS.LBL_NAME)
                opt_vars = list(set(opt_vars) - set(lbl_vars))

            momentum = 0.9
            """optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(
                loss, var_list=opt_vars)"""
            if FLAGS.OPT_TYPE == "NEST":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
            elif FLAGS.OPT_TYPE == "MOME":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=False)
            elif FLAGS.OPT_TYPE == "ADAM":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss, var_list=opt_vars)

            gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
            # accumulate
            if accum_iters != 1:
                accum_grads = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in variables]                                        
                
                zero_op = [grads.assign(tf.zeros_like(grads)) for grads in accum_grads]
                accum_op = [accum_grads[i].assign_add(g) for i, g in enumerate(gradients) if g!= None]
                avg_op = [grads.assign(grads/accum_iters) for grads in accum_grads]
            else:
                zero_op = None
                accum_op = None
                avg_op = None
                accum_grads = gradients
            if FLAGS.IS_GRAD_CLIPPING:
                """
                https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
                clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
                t_list[i] * clip_norm / max(global_norm, clip_norm),
                where global_norm = sqrt(sum([l2norm(t)**2 for t in t_list])
                if clip_norm < global_norm, the gradients will be scaled to smaller values,
                especially, if clip_norm == 1, the graidents will be normed
                """
                clipped_grads, global_norm = (
                    #tf.clip_by_global_norm(gradients) )
                    tf.clip_by_global_norm(accum_grads, clip_norm=FLAGS.GRAD_CLIPPING_NORM))
                grads_and_vars = zip(clipped_grads, variables)
            else:
                grads_and_vars = zip(accum_grads, variables)
            # global_step is incremented by one after the variables have been updated.
            op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # tensorboard
            for g, v in grads_and_vars:
                # print v.name
                name = v.name.replace(":", "_")
                #tf.summary..histogram(name+"_gradients", g)
            #tf.summary..scalar("Learning_rate", learning_rate)
            return op, (zero_op, accum_op, avg_op), reset_decay_op, learning_rate

    @lazy_property
    def prediction(self):
        return self._autoencoder_prediction

    def tf_load(self, sess, scope='autoencoder', name='deep_cae.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        path = FLAGS.AE_PATH+'/'+scope
        if not os.path.exists(path):
            print("Wrong path: {}".format(path))
        saver.restore(sess, path +'/'+name)
        print("Restore model from {}".format(path +'/'+name))

    def tf_save(self, sess, scope='autoencoder', name='deep_cae.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        path = FLAGS.AE_PATH+'/'+scope
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(sess, path +'/'+name)
        print("Save model to {}".format(path +'/'+name))
