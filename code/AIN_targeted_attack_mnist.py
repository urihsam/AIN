
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
import nn.attain_v6_mnist as attain
from utils import model_utils_mnist as model_utils
from utils.fgsm_attack import fgm
from utils.data_utils_mnist_raw import dataset

model_utils.set_flags()

# In[3]:
clean_train_saved = True
clean_test_saved = True
targeted_class_id = 8 # 0-9
data = dataset(FLAGS.DATA_DIR, split_ratio=1.0, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)

valid_frequency = 1
stop_threshold = 0.8
stop_count = 5


## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    # Placeholder nodes.
    images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
    atk_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    tgt_label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
    low_bound_holder = tf.placeholder(tf.float32, ())
    up_bound_holder = tf.placeholder(tf.float32, ())
    epsilon_holder = tf.placeholder(tf.float32, ())
    beta_x_t_holder = tf.placeholder(tf.float32, ())
    beta_x_f_holder = tf.placeholder(tf.float32, ())
    beta_y_t_holder = tf.placeholder(tf.float32, ())
    beta_y_f_holder = tf.placeholder(tf.float32, ())
    beta_y_f2_holder = tf.placeholder(tf.float32, ())
    beta_y_c_holder = tf.placeholder(tf.float32, ())
    partial_loss_holder = tf.placeholder(tf.string, ())
    is_training = tf.placeholder(tf.bool, ())

    model = attain.ATTAIN(images_holder, label_holder, low_bound_holder, up_bound_holder, 
                        epsilon_holder, is_training, targeted_label=tgt_label_holder)

    data_adv = model.prediction
    clean_acc = model._target_accuracy
    adv_acc = model._target_adv_accuracy
    #


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    # Load target classifier
    model._target.tf_load(sess, FLAGS.MNISTCNN_PATH, "target", "mnist_cnn.ckpt")
    model.tf_load(sess, name=FLAGS.AE_CKPT_RESTORE_NAME)

    total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
    total_valid_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
    total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)

    
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
            atk_path = clean_path + "/AIN_t{}".format(targeted_class_id)
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
                    targeted_label = np.expand_dims(model_utils._one_hot_encode(int(targeted_class_id), FLAGS.NUM_CLASSES), axis=0)


                    feed_dict = {
                        images_holder: clean_img, 
                        label_holder: clean_label,
                        tgt_label_holder: targeted_label,
                        low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
                        up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
                        epsilon_holder: FLAGS.EPSILON,
                        beta_x_t_holder: FLAGS.BETA_X_TRUE,
                        beta_x_f_holder: FLAGS.BETA_X_FAKE,
                        beta_y_t_holder: FLAGS.BETA_Y_TRANS,
                        beta_y_f_holder: FLAGS.BETA_Y_FAKE,
                        beta_y_f2_holder: FLAGS.BETA_Y_FAKE2,
                        beta_y_c_holder: FLAGS.BETA_Y_CLEAN,
                        is_training: False,
                        partial_loss_holder: FLAGS.PARTIAL_LOSS
                        }
                                   
                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    atk_data = sess.run(fetches=data_adv, feed_dict=feed_dict)
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
                    print("Random adv example acc:{}".format(sess.run(adv_acc, feed_dict)))
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
            atk_path = clean_path + "/AIN_t{}".format(targeted_class_id)
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
                    targeted_label = np.expand_dims(model_utils._one_hot_encode(int(targeted_class_id), FLAGS.NUM_CLASSES), axis=0)


                
                    feed_dict = {
                        images_holder: clean_img, 
                        label_holder: clean_label,
                        tgt_label_holder: targeted_label,
                        low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
                        up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
                        epsilon_holder: FLAGS.EPSILON,
                        beta_x_t_holder: FLAGS.BETA_X_TRUE,
                        beta_x_f_holder: FLAGS.BETA_X_FAKE,
                        beta_y_t_holder: FLAGS.BETA_Y_TRANS,
                        beta_y_f_holder: FLAGS.BETA_Y_FAKE,
                        beta_y_f2_holder: FLAGS.BETA_Y_FAKE2,
                        beta_y_c_holder: FLAGS.BETA_Y_CLEAN,
                        is_training: False,
                        partial_loss_holder: FLAGS.PARTIAL_LOSS
                        }               
                    #import pdb; pdb.set_trace()
                    start = time.time()
                    # c&w attack
                    atk_data = sess.run(fetches=data_adv, feed_dict=feed_dict)
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
                    print("Random adv example acc:{}".format(sess.run(adv_acc, feed_dict)))
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
        with open("AIN_targeted_info.txt", "a+") as file: 
            file.write("Train cost: {}s per example\n".format(train_cost))
            file.write("Test cost: {}s per example\n".format(test_cost))
            if train_l_count != 0.0:
                file.write("L inf distance of train adv example: {}\n".format(train_l_inf/train_l_count))
                file.write("L 2 distance of train adv example: {}\n".format(train_l_2/train_l_count))
            if test_l_count != 0.0:
                file.write("L inf distance of test adv example: {}\n".format(test_l_inf/test_l_count))
                file.write("L 2 distance of test adv example: {}\n".format(test_l_2/test_l_count))

    
    batch_xs = np.load("test_plot_clean_img.npy")
    batch_ys = np.load("test_plot_clean_y.npy")

    targeted_label = np.asarray(model_utils._one_hot_encode(
                        [int(FLAGS.TARGETED_LABEL)]*10, FLAGS.NUM_CLASSES))

    #import pdb; pdb.set_trace()
    feed_dict = {
        images_holder: batch_xs, 
        label_holder: batch_ys,
        tgt_label_holder: targeted_label
    }
    # c&w attack
    adv_images = sess.run(fetches=data_adv, feed_dict=feed_dict)

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
    new_im.save('AIN_MNIST_TGT_results.jpg')


