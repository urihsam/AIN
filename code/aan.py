from utils.attack_utils import fgm
import deep_cae as dcae
import resnet
from utils.decorator import *
from dependency import *
    

class AAN:
    """
    Adversarial Attack Network
    """

    def __init__(self, data, label, low_bound, up_bound, partial_loss, is_training):
        self.data = data
        self.label = label
        self.output_low_bound = low_bound
        self.output_up_bound = up_bound
        self.partial_loss = partial_loss
        self.is_training = is_training

        with tf.variable_scope('autoencoder'):
            #self._autoencoder = bae.BasicAE()
            self._autoencoder = dcae.DeepCAE(output_low_bound=self.output_low_bound, 
                                             output_up_bound=self.output_up_bound
                                            )
            self._autoencoder_prediction = self.data + self._autoencoder.prediction(self.data, self.is_training)
        with tf.variable_scope('target') as scope:
            self._target_adv = resnet.resnet18()
            #self._target_adv = resnet.resnet18()
            self._target_adv_logits, self._target_adv_prediction = self._target_adv.prediction(self._autoencoder_prediction)
            self._target_adv_accuracy = self._target_adv.accuracy(self._target_adv_prediction, self.label)

        with tf.variable_scope(scope, reuse=True):
            self._target_attack = resnet.resnet18()
            self.data_fake = fgm(self._target_attack.prediction, data, label, eps=FLAGS.EPSILON, iters=FLAGS.FGM_ITERS)

        with tf.variable_scope(scope, reuse=True):
            self._target_fake = resnet.resnet18()
            self._target_fake_logits, self._target_fake_prediction = self._target_fake.prediction(self.data_fake)
            self.label_fake = self.get_label(self._target_fake_prediction)
            self._target_fake_accuracy = self._target_fake.accuracy(self._target_fake_prediction, label)

        with tf.variable_scope(scope, reuse=True):
            self._target = resnet.resnet18()
            self._target_logits, self._target_prediction = self._target.prediction(data)
            self._target_accuracy = self._target.accuracy(self._target_prediction, label)
            
        self.loss_x
        self.loss_y
        self.prediction


    def get_label(self, prediction):
        n_class = prediction.get_shape().as_list()[1]
        indices = tf.argmax(prediction, axis=1)
        return tf.one_hot(indices, n_class, on_value=1.0, off_value=0.0)


    def vectorize(self, x):
        return tf.reshape(x, [-1, FLAGS.IMAGE_ROWS * FLAGS.IMAGE_COLS * FLAGS.NUM_CHANNELS])
        

    @lazy_property
    def loss_x(self):
        loss_betaX = FLAGS.BETA_X

        y_pred = self.vectorize(self._autoencoder_prediction) # adv
        y_true = self.vectorize(self.data)
        """
        Lx is for the distance loss between examples and adv_examples;
        Ly is for the prediction loss between examples and adv_examples
        """
        max_dist = tf.reduce_max(tf.abs(y_pred-y_true))
        if FLAGS.NORM_TYPE == "L2":
            #Lx_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum((y_pred-y_true)**2, 1)))
            Lx_dist = tf.reduce_mean(tf.norm(y_pred-y_true, ord=2, axis=1))
        if FLAGS.NORM_TYPE == "L1":
            #Lx_dist = tf.reduce_mean(tf.reduce_sum(tf.abs(y_pred-y_true), 1))
            Lx_dist = tf.reduce_mean(tf.norm(y_pred-y_true, ord=1, axis=1))
        elif FLAGS.NORM_TYPE == "INF":
            #Lx_dist = tf.reduce_mean(tf.reduce_max(tf.abs(y_pred-y_true), 1))]
            Lx_dist = tf.reduce_mean(tf.norm(y_pred-y_true, ord=np.inf, axis=1))
            
            
        Lx = loss_betaX * Lx_dist

        tf.summary.scalar("Loss_x", Lx)
        tf.summary.scalar("Dist_x", Lx_dist)
        tf.summary.scalar("Max_pixel_dist", max_dist)
        return Lx, Lx_dist, max_dist, (y_pred, y_true)
    

    @lazy_property
    def loss_y(self):
        loss_betaY = FLAGS.BETA_Y
        def loss_y():
            if FLAGS.LOSS_MODE == "LOGITS":
                Ly_dist = tf.reduce_mean(
                    tf.sqrt(
                        tf.reduce_sum((self._target_adv_logits-self._target_fake_logits)**2, 1)
                    )
                )
            elif FLAGS.LOSS_MODE == "PREDS":
                Ly_dist = tf.reduce_mean(
                    tf.sqrt(
                        tf.reduce_sum((self._target_adv_prediction-self._target_fake_prediction)**2, 1)
                    )
                )
            elif FLAGS.LOSS_MODE == "ENTRO":
                Ly_dist = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_fake, logits = self._target_adv_logits)
                    # -tf.reduce_sum(self.label_fake * tf.log(self._target_adv_prediction), 1)
                )
            return loss_betaY * Ly_dist, Ly_dist
        Ly, Ly_dist = loss_y()

        tf.summary.scalar("Loss_y", Ly)
        tf.summary.scalar("Dist_y", Ly_dist)
        return Ly, Ly_dist
    

    @lazy_method
    def loss(self, loss_x, loss_y):
        loss = tf.cond(tf.equal(self.partial_loss, True), lambda: loss_y, lambda: loss_x + loss_y)
        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "autoencoder")

        if FLAGS.REG_SCALE is not None:
            regularize = tf.contrib.layers.l2_regularizer(FLAGS.REG_SCALE)
            #print tf.GraphKeys.TRAINABLE_VARIABLES
            reg_term = sum([regularize(param) for param in opt_vars])
            loss += reg_term
            tf.summary.scalar("Regularization", reg_term)
        tf.summary.scalar("Total_loss", loss)
        return loss


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
