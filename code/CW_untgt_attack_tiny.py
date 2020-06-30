
# coding: utf-8

# In[1]:

import sys
import os


# In[2]:
import time
import glob
import os, math
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from dependency import *
import nn.resnet as resnet
from utils import model_utils as model_utils
from utils.cw_l2_attack import cwl2
from utils.data_utils import dataset

model_utils.set_flags()

# In[3]:
clean_saved = False
data = dataset(FLAGS.DATA_DIR, split_ratio=1.0, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_INDEX

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
    #import pdb; pdb.set_trace()
    
    with tf.variable_scope('target') as scope:
        t_model = resnet.resnet18()
        _, clean_pred = t_model.prediction(images_holder)
        clean_acc = t_model.accuracy(clean_pred, label_holder)

    with tf.variable_scope(scope, reuse=True):
        t_model = resnet.resnet18()
    cw = cwl2(t_model, model_scope=scope, dataset_type="tiny", learning_rate=1e-3, max_iterations=20000, 
        confidence=0, targeted=False, boxmin = 0.0, boxmax = 1.0, 
        scope="CW_ATTACK")
    
    with tf.variable_scope(scope, reuse=True):
        t_model = resnet.resnet18()
        _, fake_pred = t_model.prediction(fake_holder)
        fake_acc = t_model.accuracy(fake_pred, label_holder)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())

    '''
    ## distances
    size = 10 
    batch_xs, batch_ys, _ = data.next_train_batch(size, with_path=True)

    # attack
    start = time.time()
    adv_images = cw.attack(sess, batch_xs, batch_ys)
    time_cost = time.time() - start

    print("Time cost of generating per adv example: {}".format(time_cost/size))

    l_inf = np.mean(
        np.amax(
            np.absolute(np.reshape(adv_images, (size, 28*28))-np.reshape(batch_xs, (size, 28*28))), 
            axis=-1)
        )
    
    l_2 = np.mean(
        np.sqrt(np.sum(
            np.square(np.reshape(adv_images, (size, 28*28))-np.reshape(batch_xs, (size, 28*28))), 
            axis=-1)
        ))
    print("L inf: {}".format(l_inf))
    print("L 2: {}".format(l_2))

    ## acc
    feed_dict = {
        images_holder: batch_xs, 
        label_holder: batch_ys,
        fake_holder: adv_images
    }
    clean_accuracy, fake_accuracy = sess.run(fetches=[clean_acc, fake_acc], feed_dict=feed_dict)
    print("Clean accuracy: {}".format(clean_accuracy))
    print("Fake accuracy: {}".format(fake_accuracy))
    '''
    
    

    batch_xs = np.load("tiny_plot_examples.npy")/255.0
    batch_ys = np.load("tiny_plot_example_labels.npy")
    size = 10
    #import pdb; pdb.set_trace()
    start = time.time()
    adv_images = []
    for idx in range(size):
        adv_images.append(cw.attack(sess, np.expand_dims(batch_xs[idx], axis=0), np.expand_dims(batch_ys[idx], axis=0)))
    time_cost = time.time() - start
    adv_images = np.concatenate(adv_images, 0)

    
    print("Time cost of generating per adv example: {}".format(time_cost/size))

    #import pdb; pdb.set_trace()
    l_inf = np.mean(
        np.amax(
            np.absolute(np.reshape(adv_images, (size, 64*64*3))-np.reshape(batch_xs, (size, 64*64*3))), 
            axis=-1)
        )
    
    l_2 = np.mean(
        np.sqrt(np.sum(
            np.square(np.reshape(adv_images, (size, 64*64*3))-np.reshape(batch_xs, (size, 64*64*3))), 
            axis=-1)
        ))
    print("L inf: {}".format(l_inf))
    print("L 2: {}".format(l_2))

    ## acc
    feed_dict = {
        images_holder: batch_xs, 
        label_holder: batch_ys,
        fake_holder: adv_images
    }
    clean_accuracy, fake_accuracy = sess.run(fetches=[clean_acc, fake_acc], feed_dict=feed_dict)
    print("Clean accuracy: {}".format(clean_accuracy))
    print("Fake accuracy: {}".format(fake_accuracy))

    width = 10*64
    height = 2*64
    new_im = Image.new('RGB', (width, height))
    x_offset = 0
    y_offset = 64
    for i in range(10):
        im1 = Image.fromarray(np.uint8(batch_xs[i]*255.0))
        im2 = Image.fromarray(np.uint8(adv_images[i]*255.0))
        new_im.paste(im1, (x_offset, 0))
        new_im.paste(im2, (x_offset, y_offset))
        x_offset += im1.size[0]

    new_im.show()
    new_im.save('CW_TINY_UNTGT_results.jpg')


    with open("CW_tiny_untgt_statistics.txt", "a+") as file: 
        file.write("Time cost of generating per adv example: {}\n".format(time_cost/size))
        file.write("L inf: {}\n".format(l_inf))
        file.write("L 2: {}\n".format(l_2))
        file.write("Clean accuracy: {}\n".format(clean_accuracy))
        file.write("Fake accuracy: {}\n".format(fake_accuracy))
        '''
        # for 1000 itrs
        file.write("Time cost of generating per adv example: {}s\n".format(87.34737510061264))
        file.write("L inf: {}\n".format(0.6787849522819742))
        file.write("L 2: {}\n".format(2.418086041701023))
        file.write("Clean accuracy: {}\n".format(0.9959999918937683))
        file.write("Fake accuracy: {}\n".format(0.006000000052154064))
        '''

