from dependency import *
from utils.decorator import lazy_method, lazy_method_no_scope
from utils.ckpt_reader import ckpt_to_dict
from nn.resnet_model import Model


class resnet18:
    def __init__(self):
        self.model = Model(
            resnet_size=18,
            bottleneck=False,
            num_classes=200,
            num_filters=64,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=0,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=[2, 2, 2, 2],
            block_strides=[1, 2, 2, 2],
            final_size=512,
            version=2,
            data_format=None)

    @lazy_method_no_scope
    def prediction(self, inputs, use_summary=True):
        self.inputs = inputs
        """
        The structure of the network.
        """
        inputs = tf.reshape(self.inputs, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        inputs = inputs * 255.0

        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        features = inputs - tf.constant(_CHANNEL_MEANS)
        logits = self.model(features, False)
        y_conv = tf.nn.softmax(logits)

        # self.saver = tf.train.Saver()
        if use_summary:
            # tensorboard
            tf.summary.histogram("Logits", logits)
            tf.summary.histogram("Predictions", y_conv)
        return logits, y_conv

    @lazy_method_no_scope
    def loss(self, prediction, groundtruth):
        logprob = tf.log(prediction + 1e-12)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(groundtruth * logprob, axis=1))
        # tensorboard
        tf.summary.scalar("Loss", cross_entropy)
        return cross_entropy


    @lazy_method_no_scope
    def optimization(self, cross_entropy):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
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

    def tf_load(self, sess, path, name):
        """
        Load trained model from .ckpt file.
        """
        file_name = path+'/'+name
        #saver = tf.train.Saver(ckpt_to_dict(file_name))
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target'))
        self.saver.restore(sess, file_name)

    def tf_load_meta(self, sess, path, name):
        """
        Load trained model from .meta file.
        """
        file_name = path+'/'+name
        print(file_name)
        saver = tf.train.import_meta_graph(file_name)
        saver.restore(sess, file_name)


    '''def tf_save(self, sess, path, name):
        """
        Save trained model to .ckpt file.
        """
        file_name = path+'/'+name
        saver = tf.train.Saver(ckpt_to_dict(file_name))
        saver.save(sess, file_name)'''
