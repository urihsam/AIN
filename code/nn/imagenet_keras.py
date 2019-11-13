from dependency import *
from utils.decorator import lazy_method, lazy_method_no_scope, lazy_property
from nn.cnn.abccnn import ABCCNN
import nn.resnet_keras_model as resnet_keras

slim = tf.contrib.slim

class resnet50(ABCCNN):
    '''
    def __init__(self):
        self.model = tf.keras.applications.VGG16(
                    include_top=True, 
                    weights='imagenet',
                    input_shape=(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
                    )
    '''
    def __init__(self,
                 weight_path=None):
        self.weight_path = weight_path
        self.model = resnet_keras.resnet152_model(weight_path)
    

    @lazy_method
    def evaluate(self, data, is_training, use_summary=True):
        self.is_training = is_training
        """
        The structure of the network.
        """
        inputs = tf.reshape(data, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        inputs = inputs * 255.0
        means = [103.939, 116.779, 123.68]
        inputs = inputs - means
        
        logits = self.model(inputs)
        preds = tf.nn.softmax(logits, axis=1)
        # self.saver = tf.train.Saver()
        if use_summary:
            # tensorboard
            tf.summary.histogram("Logits", logits)
            tf.summary.histogram("Predictions", preds)
        return logits, preds

    
    @lazy_method
    def prediction(self, inputs, use_summary=True):
        return self.evaluate(data, False, use_summary)


    @lazy_method_no_scope
    def loss(self, logits, groundtruth, loss_type="xentropy"):
        if loss_type == "xentropy":
            '''
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=groundtruth, logits=logits
            ))
            '''
            preds = tf.nn.softmax(logits, axis=1)
            cross_entropy = -tf.reduce_sum(
                groundtruth * tf.log(1e-10+preds) + (1-groundtruth) * tf.log(1e-10+1-preds), 
                axis=1
            )
            loss = tf.reduce_mean(cross_entropy)
        elif loss_type == "mse":
            preds = tf.nn.softmax(logits, axis=1)
            loss = tf.reduce_mean(
                tf.reduce_sum((preds - groundtruth) ** 2, 1),
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

    def load_weight(self):
        # load weights
        print("Loading weights...")
        self.model.load_weights(self.weight_path, by_name=True)


    def tf_load(self, sess, path, scope, name='resnet50.ckpt', spec=""):
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)
        

    def tf_save(self, sess, path, scope, name='resnet50.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)