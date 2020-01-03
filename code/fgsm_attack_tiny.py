
# coding: utf-8

# In[1]:

import sys
import os


# In[2]:
import time
import os, math
import numpy as np
from PIL import Image
from dependency import *
import nn.resnet as resnet
from utils import model_utils
from utils.fgsm_attack import fgm
from utils.data_utils import dataset

model_utils.set_flags()

# In[3]:
data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED, split_ratio=1.0)

valid_frequency = 1
stop_threshold = 0.8
stop_count = 5


# In[5]:

## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
    with tf.variable_scope('target') as scope:
        t_model = resnet.resnet18()
        _, clean_pred = t_model.prediction(images_holder)
        clean_acc = t_model.accuracy(clean_pred, label_holder)
    with tf.variable_scope(scope, reuse=True):
        t_model = resnet.resnet18()
        data_fake = fgm(t_model.prediction, images_holder, label_holder, 
                        eps=FLAGS.EPSILON, iters=FLAGS.FGM_ITERS, clip_min=0., clip_max=1.)
    with tf.variable_scope(scope, reuse=True):
        t_model = resnet.resnet18()
        _, fake_pred = t_model.prediction(data_fake)
        fake_acc = t_model.accuracy(fake_pred, label_holder)



with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    # Load target classifier
    t_model.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
    total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
    total_valid_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
    total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    
    # training, validation and test
    for epoch in range(FLAGS.NUM_EPOCHS):
        print("Epoch {}: ".format(epoch))
        
        time_cost = 0
        time_count = 0
        for train_idx in range(total_train_batch):
            batch_xs, batch_ys, path = data.next_train_batch(FLAGS.BATCH_SIZE, with_path=True)
            #path name
            path_name = path[0].split("/")
            new_path = "/".join(path_name[:-1] +  ["fgsm"])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            file_path = "/".join(path_name[:-1] + ["fgsm", path_name[-1]])

            if not os.path.exists(file_path):
                feed_dict = {
                    images_holder: batch_xs, 
                    label_holder: batch_ys
                } 
                start = time.time()
                # c&w attack
                atk_data = sess.run(fetches=data_fake, feed_dict=feed_dict)
                time_cost += (time.time() - start)
                time_count += 1
                # save
                #im = Image.fromarray((atk_data[0]*255.0).astype(np.uint8))
                #im.save(file_path)
                np.save(file_path, atk_data[0]*255.0)
            else:
                print("fgsm adv images has existed")
        

            if train_idx % 2500 == 2499:
                print("{} adversarial examples have been generated".format(train_idx+1))
                print("Random clean example acc:{}".format(sess.run(clean_acc, feed_dict)))
                print("Random adv example acc:{}".format(sess.run(fake_acc, feed_dict)))
                print("Random adv example distance:{}".format(np.max(atk_data[0]-batch_xs[0])))
        
        if time_count != 0:
            train_cost = time_cost / time_count
        else:
            train_cost = 0
        print("Training dataset done!\n")

        time_cost = 0
        time_count = 0
        for valid_idx in range(total_valid_batch):
            batch_xs, batch_ys, path = data.next_valid_batch(FLAGS.BATCH_SIZE, with_path=True)
            # path name
            path_name = path[0].split("/")
            new_path = "/".join(path_name[:-1] +  ["fgsm"])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            file_path = "/".join(path_name[:-1] + ["fgsm", path_name[-1]])

            if not os.path.exists(file_path):
                feed_dict = {
                    images_holder: batch_xs, 
                    label_holder: batch_ys
                } 
                start = time.time()
                # c&w attack
                atk_data = sess.run(fetches=data_fake, feed_dict=feed_dict)
                time_cost += (time.time() - start)
                time_count += 1
                
                # save
                #im = Image.fromarray((atk_data[0]*255.0).astype(np.uint8))
                #im.save(file_path)
                np.save(file_path, atk_data[0]*255.0)
            else:
                print("fgsm adv images has existed")
        

            if valid_idx % 2500 == 2499:
                print("{} adversarial examples have been generated".format(valid_idx+1))
                print("Random clean example acc:{}".format(sess.run(clean_acc, feed_dict)))
                print("Random adv example acc:{}".format(sess.run(fake_acc, feed_dict)))
                print("Random adv example distance:{}".format(np.max(atk_data[0]-batch_xs[0])))
        
        if time_count != 0:
            valid_cost = time_cost / time_count
        else:
            valid_cost = 0
        print("Validation dataset done!\n")

        time_cost = 0
        time_count = 0
        for test_idx in range(total_test_batch):
            batch_xs, batch_ys, path = data.next_test_batch(FLAGS.BATCH_SIZE, with_path=True)
            # path name
            path_name = path[0].split("/")
            new_path = "/".join(path_name[:-1] +  ["fgsm"])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            file_path = "/".join(path_name[:-1] + ["fgsm", path_name[-1]])

            if not os.path.exists(file_path):
                feed_dict = {
                    images_holder: batch_xs, 
                    label_holder: batch_ys
                } 
                start = time.time()
                # c&w attack
                atk_data = sess.run(fetches=data_fake, feed_dict=feed_dict)
                time_cost += (time.time() - start)
                time_count += 1

                # save
                #im = Image.fromarray((atk_data[0]*255.0).astype(np.uint8))
                #im.save(file_path)
                np.save(file_path, atk_data[0]*255.0)
            else:
                print("fgsm adv images has existed")
        

            if test_idx % 2500 == 2499:
                print("{} adversarial examples have been generated".format(test_idx+1))
                print("Random clean example acc:{}".format(sess.run(clean_acc, feed_dict)))
                print("Random adv example acc:{}".format(sess.run(fake_acc, feed_dict)))
                print("Random adv example distance:{}".format(np.max(atk_data[0]-batch_xs[0])))
        
        if time_count != 0:
            test_cost = time_cost / time_count
        else:
            test_cost = 0
        print("Test dataset done!\n")
        print("Train cost: {}s per example".format(train_cost))
        print("Valid cost: {}s per example".format(valid_cost))
        print("Test cost: {}s per example".format(test_cost))
        with open("fgsm_tiny_info.txt", "a+") as file: 
            file.write("Train cost: {}s per example".format(train_cost))
            file.write("Valid cost: {}s per example".format(valid_cost))
            file.write("Test cost: {}s per example".format(test_cost))
        
            
        