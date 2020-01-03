
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
clean_train_saved = False
clean_test_saved = False
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
    images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
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
        data_fake = fgm(t_model.prediction, images_holder, label_holder, 
                        eps=FLAGS.EPSILON, iters=FLAGS.FGM_ITERS, clip_min=0., clip_max=1.)
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
        _, fake_pred = t_model.prediction(data_fake)
        fake_acc = t_model.accuracy(fake_pred, label_holder)


with tf.Session(graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    # Load target classifier
    t_model.tf_load(sess, FLAGS.MNISTCNN_PATH, "target", "mnist_cnn.ckpt")

    total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
    total_valid_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
    total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)

    # save training, test clean images
    if not clean_train_saved:
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
                    #clean_im = Image.fromarray(np.squeeze((batch_xs[0]*255.0).astype(np.uint8)))
                    #clean_im.save(clean_filename)
                    np.save(clean_filename, batch_xs[0]*255.0)
                
            
                if train_idx % 2500 == 2499:
                    print("{} clean examples have been saved".format(train_idx+1))
            
            print("Training dataset done!\n")

    # save test clean images
    if not clean_test_saved:
        # training, validation and test
        for epoch in range(1):
            print("Epoch {}: ".format(epoch))
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
                    #clean_im = Image.fromarray(np.squeeze((batch_xs[0]*255.0).astype(np.uint8)))
                    #clean_im.save(clean_filename)
                    np.save(clean_filename, batch_xs[0]*255.0)
                
            
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
            atk_path = clean_path + "/fgsm"
            if not os.path.exists(atk_path):
                os.mkdir(atk_path)
            
            clean_images_names = glob.glob(os.path.join(clean_path, '*.npy'))
            for clean_img_name in clean_images_names:
                path_name = clean_img_name.split("/")
                atk_img_name = os.path.join(atk_path, path_name[-1])

                if not os.path.exists(atk_img_name):
                    train_count += 1
                    #clean_im = Image.open(clean_img_name)
                    #clean_img = np.asarray(clean_im)
                    #assert clean_img.dtype == np.uint8
                    #clean_img = clean_img.astype(np.float32)
                    clean_img = np.load(clean_img_name)
                    assert clean_img.shape == (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
                    clean_img = np.expand_dims(clean_img / 255.0, axis=0)
                    clean_label = np.expand_dims(_one_hot_encode(int(class_id), FLAGS.NUM_CLASSES), axis=0)


                    feed_dict = {
                        images_holder: clean_img, 
                        label_holder: clean_label
                    }                
                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    atk_data = sess.run(fetches=data_fake, feed_dict=feed_dict)
                    time_cost += (time.time() - start)
                    # save
                    #import pdb; pdb.set_trace()
                    
                    #atk_im = Image.fromarray(np.squeeze((atk_data[0]*255.0).astype(np.uint8)))
                    #atk_im.save(atk_img_name)
                    np.save(atk_img_name, atk_data[0]*255.0)
        
                if train_count % 2500 == 2499:
                    print("{} adversarial examples have been generated".format(train_count))
                    print("Random clean example acc:{}".format(sess.run(clean_acc, feed_dict)))
                    print("Random adv example acc:{}".format(sess.run(fake_acc, feed_dict)))
                    print("Random adv example distance:{}".format(np.max(atk_data[0]-clean_img[0])))
        
        if train_count !=0:
            train_cost = time_cost / train_count
        else:
            train_cost = 0
        print("Training dataset done!\n")


        time_cost = 0
        test_count = 0
        train_path = FLAGS.DATA_DIR + "/test"
        for class_id in range(FLAGS.NUM_CLASSES):
            img_path = train_path + "/{}".format(class_id)
            clean_path = img_path + "/images"
            atk_path = clean_path + "/fgsm"
            if not os.path.exists(atk_path):
                os.mkdir(atk_path)
            
            clean_images_names = glob.glob(os.path.join(clean_path, '*.npy'))
            for clean_img_name in clean_images_names:
                path_name = clean_img_name.split("/")
                atk_img_name = os.path.join(atk_path, path_name[-1])
                if not os.path.exists(atk_img_name):
                    test_count += 1
                    #clean_im = Image.open(clean_img_name)
                    #clean_img = np.asarray(clean_im)
                    #assert clean_img.dtype == np.uint8
                    #clean_img = clean_img.astype(np.float32)
                    clean_img = np.load(clean_img_name)
                    assert clean_img.shape == (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
                    clean_img = np.expand_dims(clean_img / 255.0, axis=0)
                    clean_label = np.expand_dims(_one_hot_encode(int(class_id), FLAGS.NUM_CLASSES), axis=0)


                
                    feed_dict = {
                        images_holder: clean_img, 
                        label_holder: clean_label
                    }                
                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    atk_data = sess.run(fetches=data_fake, feed_dict=feed_dict)
                    time_cost += (time.time() - start)
                    # save
                    #import pdb; pdb.set_trace()
                    
                    #atk_im = Image.fromarray(np.squeeze((atk_data[0]*255.0).astype(np.uint8)))
                    #atk_im.save(atk_img_name)
                    np.save(atk_img_name, atk_data[0]*255.0)
            
                if test_count % 2500 == 2499:
                    print("{} adversarial examples have been generated".format(test_count))
                    print("Random clean example acc:{}".format(sess.run(clean_acc, feed_dict)))
                    print("Random adv example acc:{}".format(sess.run(fake_acc, feed_dict)))
                    print("Random adv example distance:{}".format(np.max(atk_data[0]-clean_img[0])))
        
        if test_count != 0:
            test_cost = time_cost / test_count
        else:
            test_cost = 0
        print("Test dataset done!\n")


        print("Train cost: {}s per example".format(train_cost))
        print("Test cost: {}s per example".format(test_cost))
        with open("cw_info.txt", "a+") as file: 
            file.write("Train cost: {}s per example".format(train_cost))
            file.write("Test cost: {}s per example".format(test_cost))
        
            
        