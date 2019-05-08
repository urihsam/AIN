from utils.attack_utils import fgm
import cae
import scae
import vcae
import vcae_new
import cavcae
import resnet
from utils.decorator import *
from dependency import *
    

class AAN:
    """
    Adversarial Simulation Network
    """

    def __init__(self, data, label, low_bound, up_bound, attack_epsilon, is_training):
        self.data = data
        self.label = label
        self.output_low_bound = low_bound
        self.output_up_bound = up_bound
        self.attack_epsilon = attack_epsilon
        self.is_training = is_training

        with tf.variable_scope('autoencoder'):
            if FLAGS.AE_TYPE == "VGAUSS":
                print("Using Variational Convolutional Autoencoder with Gauss")
                FLAGS.VARI = True
                self._autoencoder = vcae_new.VCAE(vtype="gauss",
                                                  output_low_bound=self.output_low_bound, 
                                                  output_up_bound=self.output_up_bound,
                                                  nonlinear_low_bound = self.output_low_bound/2.0,
                                                  nonlinear_up_bound = self.output_up_bound/2.0,
                                                  central_state_size = FLAGS.BOTTLENECK
                                                 )
            
            elif FLAGS.AE_TYPE == "TRAD":
                print("Using Convolutional Autoencoder")
                self._autoencoder = cae.CAE(output_low_bound=self.output_low_bound, 
                                            output_up_bound=self.output_up_bound,
                                            nonlinear_low_bound = self.output_low_bound/2.0,
                                            nonlinear_up_bound = self.output_up_bound/2.0,
                                            central_state_size = FLAGS.BOTTLENECK
                                           )
            
            
            self._autoencoder_prediction = self._autoencoder.prediction(self.data, self.is_training)
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
                Lx_dist = tf.reduce_mean(tf.reduce_sum(tf.square(vec1-vec2), 1))
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
    

    @lazy_method
    def loss(self, partial_loss, loss_x, loss_y):
        partial_loss_func = lambda: tf.cond(tf.equal(partial_loss, "LOSS_X"), lambda: loss_x, lambda: loss_y)
        loss = tf.cond(tf.equal(partial_loss, "FULL_LOSS"), lambda: loss_x + loss_y, partial_loss_func)
        recon_loss = FLAGS.GAMMA_R * self.loss_reconstruct
        loss += recon_loss
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
        tf.summary.scalar("Sparse_loss", sparse_loss)
        tf.summary.scalar("Variational_loss", variational_loss)
        return loss, recon_loss, sparse_loss, variational_loss, reg_term


    @lazy_method
    def optimization(self, loss):
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

        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder")

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
        op = optimizer.apply_gradients(grads_and_vars)

        # tensorboard
        for g, v in grads_and_vars:
            # print v.name
            name = v.name.replace(":", "_")
            tf.summary.histogram(name+"_gradients", g)
        tf.summary.scalar("Learning_rate", learning_rate)
        return op

    @lazy_property
    def prediction(self):
        return self._autoencoder_prediction

    def tf_load(self, sess, spec=""):
        self._autoencoder.tf_load(sess, FLAGS.AE_PATH, spec=spec)
        self._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')

    def tf_save(self, sess, spec=""):
        self._autoencoder.tf_save(sess, FLAGS.AE_PATH, spec=spec)
