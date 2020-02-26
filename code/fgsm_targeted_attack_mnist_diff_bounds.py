
# coding: utf-8

# In[1]:

import sys
import os


# In[2]:
import time
import glob
import os, math
from PIL import Image
from dependency import *
import nn.mnist_classifier as mnist_cnn
from utils import model_utils_mnist as model_utils
from utils.fgsm_attack import fgm
from utils.data_utils_mnist_raw import dataset

model_utils.set_flags()

# In[3]:
clean_train_saved = True
clean_test_saved = True
data = dataset(FLAGS.DATA_DIR, split_ratio=1.0, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)

valid_frequency = 1
stop_threshold = 0.8
stop_count = 5


## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
    fake_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    tgt_label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
    with tf.variable_scope('target') as scope:
        t_model = mnist_cnn.MNISTCNN(conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
                            conv_strides = [[2,2], [1,1], [2,2], [1,1], [2,2]], 
                            conv_channel_sizes=[16, 16, 32, 32, 64], 
                            conv_leaky_ratio=[0.2, 0.2, 0.2, 0.2, 0.2],
                            conv_drop_rate=[0.0, 0.4, 0.1, 0.2, 0.0],
                            num_res_block=1,
                            res_block_size=1,
                            res_filter_sizes=[1,1],
                            res_leaky_ratio=0.2,
                            res_drop_rate=0.2,
                            out_state=4*4*64,
                            out_fc_states=[1024, 256, 10],
                            out_leaky_ratio=0.2,
                            out_norm="NONE",
                            use_norm="NONE",
                            img_channel=1)
        _, clean_pred = t_model.prediction(images_holder)
        clean_acc = t_model.accuracy(clean_pred, label_holder)
    with tf.variable_scope(scope, reuse=True):
        t_model = mnist_cnn.MNISTCNN(conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
                            conv_strides = [[2,2], [1,1], [2,2], [1,1], [2,2]], 
                            conv_channel_sizes=[16, 16, 32, 32, 64], 
                            conv_leaky_ratio=[0.2, 0.2, 0.2, 0.2, 0.2],
                            conv_drop_rate=[0.0, 0.4, 0.1, 0.2, 0.0],
                            num_res_block=1,
                            res_block_size=1,
                            res_filter_sizes=[1,1],
                            res_leaky_ratio=0.2,
                            res_drop_rate=0.2,
                            out_state=4*4*64,
                            out_fc_states=[1024, 256, 10],
                            out_leaky_ratio=0.2,
                            out_norm="NONE",
                            use_norm="NONE",
                            img_channel=1)
        data_fake = fgm(t_model.prediction, images_holder, tgt_label_holder, 
                        eps=FLAGS.EPSILON, iters=FLAGS.FGM_ITERS, targeted=True, clip_min=0., clip_max=1.)
    with tf.variable_scope(scope, reuse=True):
        t_model = mnist_cnn.MNISTCNN(conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
                            conv_strides = [[2,2], [1,1], [2,2], [1,1], [2,2]], 
                            conv_channel_sizes=[16, 16, 32, 32, 64], 
                            conv_leaky_ratio=[0.2, 0.2, 0.2, 0.2, 0.2],
                            conv_drop_rate=[0.0, 0.4, 0.1, 0.2, 0.0],
                            num_res_block=1,
                            res_block_size=1,
                            res_filter_sizes=[1,1],
                            res_leaky_ratio=0.2,
                            res_drop_rate=0.2,
                            out_state=4*4*64,
                            out_fc_states=[1024, 256, 10],
                            out_leaky_ratio=0.2,
                            out_norm="NONE",
                            use_norm="NONE",
                            img_channel=1)
        _, fake_pred = t_model.prediction(fake_holder)
        fake_acc = t_model.accuracy(fake_pred, tgt_label_holder)


with tf.Session(graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    # Load target classifier
    t_model.tf_load(sess, FLAGS.MNISTCNN_PATH, "target", "mnist_cnn.ckpt")

    
    
    batch_xs, batch_ys = data.next_train_batch(FLAGS.BATCH_SIZE)
    tgt_ys = np.asarray(model_utils._one_hot_encode(
        [int(FLAGS.TARGETED_LABEL)]*FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES))

    #import pdb; pdb.set_trace()
    
    # attack
    start = time.time()
    adv_images = []
    for idx in range(FLAGS.BATCH_SIZE):
        feed_dict = {
            images_holder: np.expand_dims(batch_xs[idx], axis=0), 
            tgt_label_holder: np.expand_dims(tgt_ys[idx], axis=0)
        }
        adv_images.append(sess.run(fetches=data_fake, feed_dict=feed_dict))
    time_cost = time.time() - start
    print("Time cost: {}s".format(time_cost/FLAGS.BATCH_SIZE))
    adv_images = np.concatenate(adv_images, 0)

    l_inf = np.mean(
        np.amax(
            np.absolute(np.reshape(adv_images, (FLAGS.BATCH_SIZE, 28*28))-np.reshape(batch_xs, (FLAGS.BATCH_SIZE, 28*28))), 
            axis=-1)
        )
    
    l_2 = np.mean(
        np.sqrt(np.sum(
            np.square(np.reshape(adv_images, (FLAGS.BATCH_SIZE, 28*28))-np.reshape(batch_xs, (FLAGS.BATCH_SIZE, 28*28))), 
            axis=-1)
        ))
    
    print("L inf: {}".format(l_inf))
    print("L 2: {}".format(l_2))

    ## acc
    feed_dict = {
        images_holder: batch_xs, 
        label_holder: batch_ys,
        fake_holder: adv_images,
        tgt_label_holder: tgt_ys
    }
    clean_accuracy, fake_accuracy = sess.run(fetches=[clean_acc, fake_acc], feed_dict=feed_dict)
    print("Clean accuracy: {}".format(clean_accuracy))
    print("Fake accuracy: {}".format(fake_accuracy))

    with open("iFGSM_tgt_mnist_statistics_bound{}.txt".format(FLAGS.EPSILON*FLAGS.FGM_ITERS), "a+") as file: 
        file.write("Time cost: {}s\n".format(time_cost/FLAGS.BATCH_SIZE))
        file.write("L inf: {}\n".format(l_inf))
        file.write("L 2: {}\n".format(l_2))
        file.write("Clean accuracy: {}\n".format(clean_accuracy))
        file.write("Fake accuracy: {}\n".format(fake_accuracy))

    

