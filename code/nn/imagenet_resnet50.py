from dependency import *
from utils.decorator import lazy_method, lazy_method_no_scope, lazy_property
from nn.cnn.abccnn import ABCCNN
import nn.resnet_v2_model as resnet_v2
#import nn.resnet_keras_model as resnet_keras

slim = tf.contrib.slim

class resnet50(ABCCNN):
    '''
    def __init__(self,
                 # out fc layer
                 out_state=2048,
                 out_fc_states=[2048, 2048, 1000],
                 out_leaky_ratio=[0.2, 0.2, 0.2],
                 # switch
                 out_norm = None):
        self.out_state = out_state
        self.out_fc_states = out_fc_states
        self.out_leaky_ratio = out_leaky_ratio
        self.out_norm = out_norm
        self.use_class_label = False

        self.model = tf.keras.applications.ResNet50(
                include_top=False, 
                weights='imagenet',
                input_shape=(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
                )
        with tf.variable_scope("outs"):
            self.out_weight, self.out_bias = self.out_weight_bias()

    @lazy_method
    def out_weight_bias(self):
        in_size = self.out_state
        if self.use_class_label:
            in_size += FLAGS.LBL_STATES_SIZE
        state_size = self.out_fc_states
        W, b, _ = self._fc_weights_biases("W_out_", "b_out_", in_size, state_size, init_type="XV_1")
        return W, b
    

    @lazy_method
    def out_layer(self, inputs, label=None, W_name="W_out_", b_name="b_out_"):
        net = inputs
        h, w, c = net.get_shape().as_list()[1:]
        print(h, w, c)
        assert h*w*c == self.out_state
        net = tf.reshape(net, [-1, self.out_state])
        
        for layer_id in range(len(self.out_fc_states)):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.out_weight[weight_name]
            curr_bias = self.out_bias[bias_name]

            net = ne.fully_conn(net, curr_weight, curr_bias)
            if self.out_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.out_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            if layer_id != len(self.out_fc_states) - 1: # last layer
                net = ne.leaky_relu(net, self.out_leaky_ratio[layer_id])

        net = tf.identity(net, name='out_output')
        #import pdb; pdb.set_trace()
        return net
    '''

    @lazy_method_no_scope
    def evaluate(self, data, is_training, use_summary=True):
        self.is_training = is_training
        """
        The structure of the network.
        """
        inputs = tf.reshape(data, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        inputs = inputs * 255.0
        means = [123.68, 116.779, 103.939]
        inputs = inputs - means
        #inputs = tf.keras.applications.resnet50.preprocess_input(inputs)
        
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, model_info = resnet_v2.resnet_v2_50(inputs, num_classes=FLAGS.NUM_CLASSES+1, 
                                                        is_training=self.is_training)
            preds = model_info["predictions"]
        '''
        
        net = self.model(inputs)
        
        logits = self.out_layer(net)
        preds = tf.nn.softmax(logits, 1)
        '''
        # self.saver = tf.train.Saver()
        if use_summary:
            # tensorboard
            tf.summary.histogram("Logits", logits)
            tf.summary.histogram("Predictions", preds)
        return logits, preds

    
    @lazy_method_no_scope
    def prediction(self, inputs, use_summary=True):
        return self.evaluate(data, False, use_summary)

    @lazy_method_no_scope
    def loss(self, logits, groundtruth, loss_type="xentropy"):
        if loss_type == "xentropy":
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=groundtruth, logits=logits
            ))
        elif loss_type == "mse":
            loss = tf.reduce_mean(
                tf.reduce_sum((tf.nn.softmax(logits, 1) - groundtruth) ** 2, 1),
                0)
        elif loss_type == "categ":
            preds = tf.nn.softmax(logits, axis=1)
            categ = tf.keras.losses.categorical_crossentropy(
                groundtruth, preds)
            loss = tf.reduce_mean(categ)
        # tensorboard
        tf.summary.scalar("Loss", loss)
        return loss

    @lazy_method
    def optimization(self, learning_rate, loss):
        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target")
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=opt_vars)
        return optimizer


    @lazy_method_no_scope
    def accuracy(self, prediction, groundtruth):
        correct_prediction = tf.equal(
            tf.argmax(groundtruth, 1),
            tf.argmax(prediction, 1)
        )
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tensorboard
        tf.summary.scalar("Accuracy", acc)
        return acc

    def tf_load(self, sess, path, scope, name='resnet50.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='resnet50.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)