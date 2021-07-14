
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
from utils.data_utils_mnist import dataset

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
    cw = cwl2(t_model, model_scope=scope, dataset_type="mnist", learning_rate=1e-3, max_iterations=20000, 
        confidence=0, targeted=False, boxmin = 0.0, boxmax = 1.0, 
        scope="CW_ATTACK")
    
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
        _, fake_pred = t_model.prediction(fake_holder)
        fake_acc = t_model.accuracy(fake_pred, label_holder)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    # Load target classifier
    t_model.tf_load(sess, FLAGS.MNISTCNN_PATH, "target", "mnist_cnn.ckpt")
    # Adv training and test
    for epoch in range(1):
        print("Epoch {}: ".format(epoch))
        
        train_l_inf = 0.0
        train_l_2 = 0.0
        train_l_count = 0
        time_cost = 0
        train_count = 0
        train_path = FLAGS.DATA_DIR + "/train"
        for class_id in range(FLAGS.NUM_CLASSES):
            img_path = train_path + "/{}".format(class_id)
            clean_path = img_path + "/images"
            atk_path = clean_path + "/cw"
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
                    clean_label = np.expand_dims(model_utils._one_hot_encode(int(class_id), FLAGS.NUM_CLASSES), axis=0)
                    #targeted_label = np.expand_dims(model_utils._one_hot_encode(int(targeted_class_id), FLAGS.NUM_CLASSES), axis=0)

          
                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    
                    atk_data = cw.attack(sess, clean_img, clean_label)

                    time_cost += (time.time() - start)
                    # save
                    #import pdb; pdb.set_trace()
                    
                    #atk_im = Image.fromarray(np.squeeze((atk_data[0]*255.0).astype(np.uint8)))
                    #atk_im.save(atk_img_name)
                    np.save(atk_img_name, atk_data[0]*255.0)
                else:
                    #import pdb; pdb.set_trace()
                    train_l_count += 1
                    clean_img = np.reshape(np.load(clean_img_name), (784))
                    atk_img = np.reshape(np.load(atk_img_name), (784))
                    train_l_inf += np.mean(
                        np.amax(
                            np.absolute(atk_img/255.0-clean_img/255.0), axis=-1)
                        )
                    

                    train_l_2 += np.mean(
                        np.sqrt(np.sum(
                            np.square(atk_img/255.0-clean_img/255.0), axis=-1)
                        ))
                    
        
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

        test_l_2 = 0.0
        test_l_inf = 0.0
        test_l_count = 0
        time_cost = 0
        test_count = 0
        train_path = FLAGS.DATA_DIR + "/test"
        for class_id in range(FLAGS.NUM_CLASSES):
            img_path = train_path + "/{}".format(class_id)
            clean_path = img_path + "/images"
            atk_path = clean_path + "/fgsm_t{}".format(targeted_class_id)
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
                    clean_label = np.expand_dims(model_utils._one_hot_encode(int(class_id), FLAGS.NUM_CLASSES), axis=0)
                    #targeted_label = np.expand_dims(model_utils._one_hot_encode(int(targeted_class_id), FLAGS.NUM_CLASSES), axis=0)

                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    atk_data = cw.attack(sess, clean_img, clean_label)
                    time_cost += (time.time() - start)
                    # save
                    #import pdb; pdb.set_trace()
                    
                    #atk_im = Image.fromarray(np.squeeze((atk_data[0]*255.0).astype(np.uint8)))
                    #atk_im.save(atk_img_name)
                    np.save(atk_img_name, atk_data[0]*255.0)
                else:
                    test_l_count += 1
                    clean_img = np.reshape(np.load(clean_img_name), (784))
                    atk_img = np.reshape(np.load(atk_img_name), (784))
                    test_l_inf += np.mean(
                        np.amax(
                            np.absolute(atk_img/255.0-clean_img/255.0), axis=-1)
                        )
                    

                    test_l_2 += np.mean(
                        np.sqrt(np.sum(
                            np.square(atk_img/255.0-clean_img/255.0), axis=-1)
                        ))
                    
            
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
        if train_l_count != 0.0:
            print("L inf distance of train adv example: {}".format(train_l_inf/train_l_count))
            print("L 2 distance of train adv example: {}".format(train_l_2/train_l_count))
        if test_l_count != 0.0:
            print("L inf distance of test adv example: {}".format(test_l_inf/test_l_count))
            print("L 2 distance of test adv example: {}".format(test_l_2/test_l_count))
        with open("fgsm_targeted_info.txt", "a+") as file: 
            file.write("Train cost: {}s per example\n".format(train_cost))
            file.write("Test cost: {}s per example\n".format(test_cost))
            if train_l_count != 0.0:
                file.write("L inf distance of train adv example: {}\n".format(train_l_inf/train_l_count))
                file.write("L 2 distance of train adv example: {}\n".format(train_l_2/train_l_count))
            if test_l_count != 0.0:
                file.write("L inf distance of test adv example: {}\n".format(test_l_inf/test_l_count))
                file.write("L 2 distance of test adv example: {}\n".format(test_l_2/test_l_count))
    
    
    
    

    batch_xs = np.load("mnist_plot_examples.npy")/255.0
    batch_ys = np.load("mnist_plot_example_labels.npy")
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

    width = 10*28
    height = 2*28
    new_im = Image.new('L', (width, height))
    x_offset = 0
    y_offset = 28
    for i in range(10):
        im1 = Image.fromarray(np.reshape((batch_xs[i] * 255).astype(np.uint8), (28,28)))
        im2 = Image.fromarray(np.reshape((adv_images[i]*255).astype(np.uint8), (28, 28)))
        new_im.paste(im1, (x_offset, 0))
        new_im.paste(im2, (x_offset, y_offset))
        x_offset += im1.size[0]

    new_im.show()
    new_im.save('CW_MNIST_UNTGT_results.jpg')


    with open("CW_mnist_statistics.txt", "a+") as file: 
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

