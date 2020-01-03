
# coding: utf-8

# In[1]:

import sys
import os


# In[2]:
import time
import os, math
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from dependency import *
import nn.resnet as resnet
from utils import model_utils
from utils.cw_l2_attack import cwl2
from utils.data_utils import dataset

model_utils.set_flags()

# In[3]:
data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)

valid_frequency = 1
stop_threshold = 0.8
stop_count = 5


# In[5]:

## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    with tf.variable_scope('target') as scope:
        t_model = resnet.resnet18()
    cw = cwl2(t_model, model_scope=scope, max_iterations=1000, confidence=0, boxmin = 0.0, boxmax = 1.0, scope="CW_ATTACK")



with tf.Session(graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    #model.tf_load(sess, "./models/target_classifier/resnet_v2_50/resnet_v2_50_2017", "target", name='resnet_v2_50.ckpt')
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
            new_path = "/".join(path_name[:-1] +  ["cw"])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            file_path = "/".join(path_name[:-1] + ["cw", path_name[-1]])

            if not os.path.exists(file_path):
                start = time.time()
                # c&w attack
                atk_data = cw.attack(sess, batch_xs, batch_ys)
                time_cost += (time.time() - start)
                time_count += 1
                # save
                im = Image.fromarray((atk_data[0]*255.0).astype(np.uint8))
                im.save(file_path)
            else:
                print("CW adv images has existed")
        

            if train_idx % 2500 == 2499:
                print("{} adversarial examples have been generated".format(train_idx+1))
        
        train_cost = time_cost / time_count
        printf("Training dataset done!\n")

        time_cost = 0
        time_count = 0
        for valid_idx in range(total_valid_batch):
            batch_xs, batch_ys, path = data.next_valid_batch(FLAGS.BATCH_SIZE, with_path=True)
            # path name
            path_name = path[0].split("/")
            new_path = "/".join(path_name[:-1] +  ["cw"])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            file_path = "/".join(path_name[:-1] + ["cw", path_name[-1]])

            if not os.path.exists(file_path):
                # c&w attack
                start = time.time()
                atk_data = cw.attack(sess, batch_xs, batch_ys)
                time_cost += (time.time() - start)
                time_count += 1
                
                # save
                im = Image.fromarray((atk_data[0]*255.0).astype(np.uint8))
                im.save(file_path)
            else:
                print("CW adv images has existed")
        

            if valid_idx % 2500 == 2499:
                print("{} adversarial examples have been generated".format(valid_idx+1))
        
        valid_cost = time_cost / time_count
        printf("Validation dataset done!\n")

        time_cost = 0
        time_count = 0
        for test_idx in range(total_test_batch):
            batch_xs, batch_ys, path = data.next_test_batch(FLAGS.BATCH_SIZE, with_path=True)
            # path name
            path_name = path[0].split("/")
            new_path = "/".join(path_name[:-1] +  ["cw"])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            file_path = "/".join(path_name[:-1] + ["cw", path_name[-1]])

            if not os.path.exists(file_path):
                # c&w attack
                start = time.time()
                atk_data = cw.attack(sess, batch_xs, batch_ys)
                time_cost += (time.time() - start)
                time_count += 1

                # save
                im = Image.fromarray((atk_data[0]*255.0).astype(np.uint8))
                im.save(file_path)
            else:
                print("CW adv images has existed")
        

            if test_idx % 2500 == 2499:
                print("{} adversarial examples have been generated".format(test_idx+1))
        
        test_cost = time_cost / time_count
        printf("Test dataset done!\n")
        printf("Train cost: {}s per example".format(train_cost))
        printf("Valid cost: {}s per example".format(valid_cost))
        printf("Test cost: {}s per example".format(test_cost))
        with open("cw_info.txt", "a+") as file: 
            file.write("Train cost: {}s per example".format(train_cost))
            file.write("Valid cost: {}s per example".format(valid_cost))
            file.write("Test cost: {}s per example".format(test_cost))
        
            
        