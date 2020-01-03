
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
import nn.mnist_classifier as mnist_cnn
from utils import model_utils_mnist as model_utils
from utils.cw_l2_attack import cwl2
from utils.data_utils_mnist_raw import dataset

model_utils.set_flags()

# In[3]:
clean_saved = False
data = dataset(FLAGS.DATA_DIR, split_ratio=1.0, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)

valid_frequency = 1
stop_threshold = 0.8
stop_count = 5

def _one_hot_encode(inputs, encoded_size):
    def get_one_hot(number):
        on_hot=[0]*encoded_size
        on_hot[int(number)]=1
        return on_hot
    #return list(map(get_one_hot, inputs))
    if isinstance(inputs, list):
        return list(map(get_one_hot, inputs))
    else:
        return get_one_hot(inputs)
# In[5]:

## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
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
    cw = cwl2(t_model, model_scope=scope, dataset_type="mnist", max_iterations=1000, confidence=0, boxmin = 0.0, boxmax = 1.0, scope="CW_ATTACK")



with tf.Session(graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
    total_valid_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
    total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)

    # save training, test clean images
    if not clean_saved:
        # training, validation and test
        for epoch in range(1):
            print("Epoch {}: ".format(epoch))
            
            train_path = FLAGS.DATA_DIR + "/train"
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            for train_idx in range(total_train_batch):
                batch_xs, batch_ys = data.next_train_batch(FLAGS.BATCH_SIZE)
                img_path = train_path + "/{}".format(np.argmax(batch_ys[0]))
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                clean_path = img_path + "/images"
                
                if not os.path.exists(clean_path):
                    os.mkdir(clean_path)
                
                clean_filename = clean_path+"/{}.JPEG".format(train_idx)
                
                if not os.path.exists(clean_filename):
                    clean_im = Image.fromarray(np.squeeze((batch_xs[0]*255.0).astype(np.uint8)))
                    clean_im.save(clean_filename)
                
            
                if train_idx % 2500 == 2499:
                    print("{} clean examples have been saved".format(train_idx+1))
            
            print("Training dataset done!\n")


            test_path = FLAGS.DATA_DIR + "/test"
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            for test_idx in range(total_test_batch):
                batch_xs, batch_ys = data.next_test_batch(FLAGS.BATCH_SIZE)
                img_path = test_path + "/{}".format(np.argmax(batch_ys[0]))
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                
                clean_path = img_path + "/images"
                if not os.path.exists(clean_path):
                    os.mkdir(clean_path)
                
                clean_filename = clean_path+"/{}.JPEG".format(test_idx)
                if not os.path.exists(clean_filename):
                    clean_im = Image.fromarray(np.squeeze((batch_xs[0]*255.0).astype(np.uint8)))
                    clean_im.save(clean_filename)
                
            
                if test_idx % 2500 == 2499:
                    print("{} clean examples have been saved".format(test_idx+1))
            print("Test dataset done!\n")

    
    # Adv training and test
    for epoch in range(1):
        print("Epoch {}: ".format(epoch))
        
        time_cost = 0
        train_count = 0
        train_path = FLAGS.DATA_DIR + "/train"
        for class_id in range(FLAGS.NUM_CLASSES):
            img_path = train_path + "/{}".format(class_id)
            clean_path = img_path + "/images"
            atk_path = clean_path + "/cw"
            if not os.path.exists(atk_path):
                os.mkdir(atk_path)
            
            clean_images_names = glob.glob(os.path.join(clean_path, '*.JPEG'))
            for clean_img_name in clean_images_names:
                path_name = clean_img_name.split("/")
                atk_img_name = os.path.join(atk_path, path_name[-1])

                if not os.path.exists(atk_img_name):
                    train_count += 1
                    clean_im = Image.open(clean_img_name)
                    clean_img = np.asarray(clean_im)
                    assert clean_img.dtype == np.uint8
                    clean_img = clean_img.astype(np.float32)
                    clean_img = np.expand_dims(clean_img, axis=2)
                    assert clean_img.shape == (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
                    clean_img = np.expand_dims(clean_img / 255.0, axis=0)
                    clean_label = np.expand_dims(_one_hot_encode(int(class_id), FLAGS.NUM_CLASSES), axis=0)


                
                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    atk_data = cw.attack(sess, clean_img, clean_label)
                    time_cost += (time.time() - start)
                    # save
                    #import pdb; pdb.set_trace()
                    
                    atk_im = Image.fromarray(np.squeeze((atk_data[0]*255.0).astype(np.uint8)))
                    atk_im.save(atk_img_name)
        
            if train_count % 2500 == 2499:
                print("{} adversarial examples have been generated".format(train_count))
        
        train_cost = time_cost / train_count
        print("Training dataset done!\n")


        time_cost = 0
        test_count = 0
        train_path = FLAGS.DATA_DIR + "/test"
        for class_id in range(FLAGS.NUM_CLASSES):
            img_path = train_path + "/{}".format(class_id)
            clean_path = img_path + "/images"
            atk_path = clean_path + "/cw"
            if not os.path.exists(atk_path):
                os.mkdir(atk_path)
            
            clean_images_names = glob.glob(os.path.join(clean_path, '*.JPEG'))
            for clean_img_name in clean_images_names:
                path_name = clean_img_name.split("/")
                atk_img_name = os.path.join(atk_path, path_name[-1])
                if not os.path.exists(atk_img_name):
                    test_count += 1
                    clean_im = Image.open(clean_img_name)
                    clean_img = np.asarray(clean_im)
                    assert clean_img.dtype == np.uint8
                    clean_img = clean_img.astype(np.float32)
                    assert clean_img.shape == (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
                    clean_img = np.expand_dims(clean_img / 255.0, axis=0)
                    clean_label = np.expand_dims(_one_hot_encode(int(class_id), FLAGS.NUM_CLASSES), axis=0)


                
                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    atk_data = cw.attack(sess, clean_img, clean_label)
                    time_cost += (time.time() - start)
                    # save
                    #import pdb; pdb.set_trace()
                    
                    atk_im = Image.fromarray(np.squeeze((atk_data[0]*255.0).astype(np.uint8)))
                    atk_im.save(atk_img_name)
            
            if test_count % 2500 == 2499:
                print("{} adversarial examples have been generated".format(test_count))
        
        test_cost = time_cost / test_count
        print("Test dataset done!\n")


        print("Train cost: {}s per example".format(train_cost))
        print("Test cost: {}s per example".format(test_cost))
        with open("cw_info.txt", "a+") as file: 
            file.write("Train cost: {}s per example".format(train_cost))
            file.write("Test cost: {}s per example".format(test_cost))
        
            
        