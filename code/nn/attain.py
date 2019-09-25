from utils.attack_utils import fgm
import nn.cae as cae
import nn.scae as scae
import nn.vcae as vcae
import nn.vcae_new as vcae_new
import nn.cavcae as cavcae
import nn.resnet as resnet
import nn.ae.attresenc as attresenc
import nn.ae.attresdec as attresdec
#import nn.embedder_v2 as embedder
#import nn.embedder_v3 as embedder
import nn.embedder as embedder
from utils.decorator import *
from dependency import *
import math
    

class ATTAIN:
    """
    Attentive Adversarial Imitation Network
    """

    def __init__(self, data, label, low_bound, up_bound, attack_epsilon, is_training):
        self.data = data
        self.label = label
        self.output_low_bound = low_bound
        self.output_up_bound = up_bound
        self.central_emb_size = FLAGS.EMB_SIZE
        self.central_channel_size = FLAGS.CENTRAL_CHANNEL_SIZE
        self.attack_epsilon = attack_epsilon
        self.is_training = is_training

        with tf.variable_scope('autoencoder'):
            if FLAGS.AE_TYPE == "ATTAE":
                
                # Class label autoencoder
                with tf.variable_scope(FLAGS.LBL_NAME) as lbl_scope:
                    row = self.data.get_shape().as_list()[1]#data.get_shape().as_list()[1]
                    col = self.data.get_shape().as_list()[2]#int(math.ceil(FLAGS.NUM_CLASSES/data.get_shape().as_list()[1]))
                    cha = self.data.get_shape().as_list()[3]
                    img_shape = [row, col, cha]
                    size = np.prod(img_shape)
                    w1_shape = [label.get_shape().as_list()[-1], size]
                    w1 = ne.weight_variable(w1_shape, "lbl_W1", init_type="HE")
                    b1 = ne.bias_variable([size], "lbl_b1")
                    label_states = ne.sigmoid(ne.fully_conn(label, w1, b1))
                    #
                    w2_shape = [size, label.get_shape().as_list()[-1]]
                    w2 = ne.weight_variable(w2_shape, "lbl_W2", init_type="HE")
                    b2 = ne.bias_variable([label.get_shape().as_list()[-1]], "lbl_b2")
                    self._recon_label = ne.sigmoid(ne.fully_conn(label_states, w2, b2))
                    #
                    '''
                    self._label_states = []
                    
                    for _ in range(FLAGS.NUM_CHANNELS):
                        self._label_states.append(tf.reshape(label_states, [-1] + img_shape))
                    self._label_states = tf.concat(self._label_states, -1) # [B, 64, 4, 3]
                    '''
                    self._label_states = tf.reshape(label_states, [-1] + img_shape)
                    self.labeled_data = tf.multiply(self.data, self._label_states)
                    #self.labeled_data = tf.concat([self.labeled_data, self._label_states], -1)
                    

                    #self._labeled_data= tf.concat([data, self._label_states], -2) #[B, 64, 68, 3]
                    #self._labeled_central_states= tf.concat([self._central_states, self._label_states], -1)
                self.lbl_scope = lbl_scope


                # Encoder
                with tf.variable_scope(FLAGS.ENC_NAME) as enc_scope:
                    self._encoder = attresenc.ATTRESENC(
                        attention_type=FLAGS.ATT_TYPE,
                        att_pos_idx=0,
                        att_f_channel_size = 64,
                        att_g_channel_size = 64,
                        att_h_channel_size = 64,
                        conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3]],
                        conv_strides = [[2,2], [1,1], [2,2], [1,1]], 
                        conv_channel_sizes=[64, 256, 512, 1024], 
                        conv_leaky_ratio=[0.2, 0.2, 0.2, 0.2],
                        num_res_block=FLAGS.NUM_ENC_RES_BLOCK,
                        res_block_size=FLAGS.ENC_RES_BLOCK_SIZE,
                        res_filter_sizes=[1,1],
                        res_leaky_ratio=0.2,
                        out_channel_size=self.central_channel_size,
                        out_norm=FLAGS.ENC_OUT_NORM,
                        use_norm=FLAGS.ENC_NORM,
                        img_channel=FLAGS.NUM_CHANNELS)
                    self._central_states = self._encoder.evaluate(self.labeled_data, self.is_training)
                
                '''
                # Embedder
                with tf.variable_scope(FLAGS.EMB_NAME) as emb_scope:
                    self._dec_embedder = embedder.EMBEDDER(self._central_states.get_shape().as_list()[1:], 
                                                    [self.central_emb_size],
                                                    emb_norm=FLAGS.EMB_NORM)
                    self._central_embedded, self._embed_k_vs, self._embed_k_coefs =\
                        self._dec_embedder.evaluate(self._central_states, self.is_training, res_start=1)
                '''

                # Decoder
                with tf.variable_scope(FLAGS.DEC_NAME) as dec_scope:
                    self._decoder_t = attresdec.ATTRESDEC(
                        self.output_low_bound, self.output_up_bound,
                        attention_type=FLAGS.ATT_TYPE,
                        att_pos_idx=-1,
                        att_f_channel_size = 64,
                        att_g_channel_size = 64,
                        att_h_channel_size = 64,
                        decv_filter_sizes=[[3,3], [4,4], [3,3], [4,4]],
                        decv_strides = [[1,1], [2,2], [1,1], [2,2]], 
                        decv_channel_sizes=[1024, 512, 256, 64], 
                        decv_leaky_ratio=[0.2, 0.2, 0.2, 0.2],
                        num_res_block=FLAGS.NUM_DEC_RES_BLOCK,
                        res_block_size=FLAGS.DEC_RES_BLOCK_SIZE,
                        res_filter_sizes=[1,1],
                        res_leaky_ratio=0.2,
                        in_channel_size=self.central_channel_size,
                        in_norm=FLAGS.DEC_IN_NORM,
                        use_norm=FLAGS.DEC_NORM)
                    self._generated_t = self._decoder_t.evaluate(self._central_states, self.is_training)
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
            #self._target_adv = resnet.resnet18()
            self._target_adv_logits, self._target_adv_prediction = self._target_adv.prediction(self._autoencoder_prediction)
            self._target_adv_accuracy = self._target_adv.accuracy(self._target_adv_prediction, self.label)
        # Fake data generated by attacking algo
        with tf.variable_scope(scope, reuse=True):
            self._target_attack = resnet.resnet18()
            self.data_fake = fgm(self._target_attack.prediction, data, label, eps=self.attack_epsilon, iters=FLAGS.FGM_ITERS,
                                 clip_min=0., clip_max=1.)
        
        with tf.variable_scope(scope, reuse=True):
            self._target_fake = resnet.resnet18()
            self._target_fake_logits, self._target_fake_prediction = self._target_fake.prediction(self.data_fake)
            self.label_fake = self.get_label(self._target_fake_prediction)
            self._target_fake_accuracy = self._target_fake.accuracy(self._target_fake_prediction, label)
        # Clean data
        with tf.variable_scope(scope, reuse=True):
            self._target = resnet.resnet18()
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
        Lx_fake = beta_f * Lx_dist_fake

        Lx = Lx_true + Lx_fake

        tf.summary.scalar("Loss_x", Lx)
        tf.summary.scalar("Loss_x_true", Lx_true)
        tf.summary.scalar("Loss_x_fake", Lx_fake)
        tf.summary.scalar("Dist_x_true", Lx_dist_true)
        tf.summary.scalar("Dist_x_fake", Lx_dist_fake)
        tf.summary.scalar("Max_pixel_dist_true", max_dist_true)
        tf.summary.scalar("Max_pixel_dist_fake", max_dist_fake)
        return Lx, (Lx_true, Lx_fake), (Lx_dist_true, Lx_dist_fake), (max_dist_true, max_dist_fake)


    @lazy_property
    def max_distance(self):
        x_adv = self.vectorize(self._autoencoder_prediction) # adv
        x_true = self.vectorize(self.data)
        x_fake = self.vectorize(self.data_fake)
        
        max_dist_true = tf.reduce_max(tf.abs(x_adv-x_true))
        max_dist_fake = tf.reduce_max(tf.abs(x_adv-x_fake))

        tf.summary.scalar("Max_dist_true", max_dist_true)
        tf.summary.scalar("Max_dist_fake", max_dist_fake)
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

        tf.summary.scalar("Norm_dist_adv_true", norm_dist_adv_true)
        tf.summary.scalar("Norm_dist_adv_fake", norm_dist_adv_fake)
        return norm_dist_adv_true, norm_dist_adv_fake
    

    @lazy_method
    def loss_y(self, beta_t, beta_f, beta_c, kappa_t, kappa_f, kappa_c):
        def loss_y_from_trans(): 
            if FLAGS.LOSS_MODE_TRANS == "C&W": # Make the logits at the largest prob position of fake data larger than that at true position
                y_faked = tf.argmax(self._target_fake_logits, axis=1, output_type=tf.int32)
                mask1 = tf.one_hot(y_faked, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
                y_clean = tf.argmax(self.label, axis=1, output_type=tf.int32)
                mask2 = tf.one_hot(y_clean, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
                adv_logits_at_y_faked = tf.reduce_max(tf.subtract(self._target_adv_logits, mask1), axis=1)
                adv_logits_at_y_clean = tf.reduce_max(tf.subtract(self._target_adv_logits, mask2), axis=1)
                Ly_dist = tf.reduce_mean(
                    tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_at_y_faked), -kappa_t)
                )
            return beta_t * Ly_dist, Ly_dist


        def loss_y_from_fake(): 
            if FLAGS.LOSS_MODE_FAKE == "LOGITS": # Minimize the distance of logits from faked data
                Ly_dist = tf.reduce_mean(
                    # tf.sqrt(tf.reduce_sum(tf.square(self._target_adv_logits-self._target_fake_logits), 1))
                    tf.norm(self._target_adv_logits-self._target_fake_logits, ord=2, axis=1)
                )
            elif FLAGS.LOSS_MODE_FAKE == "PREDS": # Minimize the distance of prediction from faked data
                Ly_dist = tf.reduce_mean(
                    # tf.sqrt(tf.reduce_sum(tf.square(self._target_adv_prediction-self._target_fake_prediction), 1))
                    tf.norm(self._target_adv_prediction-self._target_fake_prediction, ord=2, axis=1)
                )
            elif FLAGS.LOSS_MODE_FAKE == "ENTRO": # Minimize the distance of prediction from faked data
                Ly_dist = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_fake, logits = self._target_adv_logits)
                    # -tf.reduce_sum(self.label_fake * tf.log(self._target_adv_prediction), 1)
                )
            elif FLAGS.LOSS_MODE_FAKE == "C&W": # Maximize the logits at max prob position of faked data
                y_faked = tf.argmax(self._target_fake_logits, axis=1, output_type=tf.int32)
                mask1 = tf.one_hot(y_faked, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
                mask2 = tf.one_hot(y_faked, FLAGS.NUM_CLASSES, on_value=float('inf'), off_value=0.0)
                adv_logits_at_y_faked = tf.reduce_max(tf.subtract(self._target_adv_logits, mask1), axis=1)
                adv_logits_max_exclude_y_faked = tf.reduce_max(tf.subtract(self._target_adv_logits, mask2), axis=1)
                Ly_dist = tf.reduce_mean(
                    tf.maximum(tf.subtract(adv_logits_max_exclude_y_faked, adv_logits_at_y_faked), -kappa_f)
                )
            return beta_f * Ly_dist, Ly_dist
        
        def loss_y_from_clean(): 
            if FLAGS.LOSS_MODE_CLEAN == "LOGITS": # Maximize the distance of logits from clean data
                Ly_dist = -1.0 * tf.reduce_mean(
                    # tf.sqrt(tf.reduce_sum(tf.square(self._target_adv_logits-self._target_logits), 1))
                    tf.norm(self._target_adv_logits-self._target_logits, ord=2, axis=1)
                )
            elif FLAGS.LOSS_MODE_CLEAN == "PREDS": # Maximize the distance of prediction from clean data
                Ly_dist = -1.0 * tf.reduce_mean(
                    # tf.sqrt(tf.reduce_sum(tf.square(self._target_adv_prediction-self._target_prediction), 1))
                    tf.norm(self._target_adv_prediction-self._target_prediction, ord=2, axis=1)
                )
            elif FLAGS.LOSS_MODE_CLEAN == "ENTRO": # Maximize the distance of prediction from clean data
                Ly_dist = -1.0 * tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label, logits = self._target_adv_logits)
                    # -tf.reduce_sum(self.label_fake * tf.log(self._target_adv_prediction), 1)
                )
            elif FLAGS.LOSS_MODE_CLEAN == "C&W": # Minimize the logits at clean label
                y_clean = tf.argmax(self.label, axis=1, output_type=tf.int32)
                mask1 = tf.one_hot(y_clean, FLAGS.NUM_CLASSES, on_value=0.0, off_value=float('inf'))
                mask2 = tf.one_hot(y_clean, FLAGS.NUM_CLASSES, on_value=float('inf'), off_value=0.0)
                adv_logits_at_y_clean = tf.reduce_max(tf.subtract(self._target_adv_logits, mask1), axis=1)
                adv_logits_max_exclude_y_clean = tf.reduce_max(tf.subtract(self._target_adv_logits, mask2), axis=1)
                Ly_dist = tf.reduce_mean(
                    tf.maximum(tf.subtract(adv_logits_at_y_clean, adv_logits_max_exclude_y_clean), -kappa_c)
                )
                
            return beta_c * Ly_dist, Ly_dist

        Ly_trans, Ly_dist_trans = loss_y_from_trans()
        Ly_fake, Ly_dist_fake = loss_y_from_fake()
        Ly_clean, Ly_dist_clean = loss_y_from_clean()
        Ly = Ly_trans + Ly_fake + Ly_clean

        tf.summary.scalar("Loss_y_trans", Ly_trans)
        tf.summary.scalar("Dist_y_trans", Ly_dist_trans)
        tf.summary.scalar("Loss_y_fake", Ly_fake)
        tf.summary.scalar("Dist_y_fake", Ly_dist_fake)
        tf.summary.scalar("Loss_y_clean", Ly_clean)
        tf.summary.scalar("Dist_y_clean", Ly_dist_clean)
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
        cross_entropy = -tf.reduce_sum(
            self.label * tf.log(1e-5+self._recon_label) + (1-self.label) * tf.log(1e-5 + 1-self._recon_label),
            axis = 1
        )
        return tf.reduce_mean(cross_entropy)

    
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
            regularize = tf.contrib.layers.l2_regularizer(FLAGS.REG_SCALE)
            #print tf.GraphKeys.TRAINABLE_VARIABLES
            reg_term = sum([regularize(param) for param in opt_vars])
            loss += reg_term
            tf.summary.scalar("Regularization", reg_term)
        tf.summary.scalar("Total_loss", loss)
        tf.summary.scalar("Reconstruct_loss", recon_loss)
        tf.summary.scalar("Label_loss", label_loss)
        tf.summary.scalar("Sparse_loss", sparse_loss)
        tf.summary.scalar("Variational_loss", variational_loss)
        return loss, recon_loss, label_loss, sparse_loss, variational_loss, reg_term


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
            # decay
            learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE, global_step, 
                FLAGS.LEARNING_DECAY_STEPS, FLAGS.LEARNING_DECAY_RATE)

            if scope == "PRE_OPT":
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.lbl_scope.name)
            else:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder")
                lbl_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.lbl_scope.name)
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
            op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # tensorboard
            for g, v in grads_and_vars:
                # print v.name
                name = v.name.replace(":", "_")
                tf.summary.histogram(name+"_gradients", g)
            tf.summary.scalar("Learning_rate", learning_rate)
            return op, (zero_op, accum_op, avg_op), learning_rate

    @lazy_property
    def prediction(self):
        return self._autoencoder_prediction

    def tf_load(self, sess, name='deep_cae.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.restore(sess, FLAGS.AE_PATH+'/'+name)

    def tf_save(self, sess, name='deep_cae.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.save(sess, FLAGS.AE_PATH+'/'+name)
