
# coding: utf-8

# In[1]:


import sys
import os
#PATH = "../"
import time

# In[2]:

from dependency import *
import nn.resnet as resnet
from PIL import Image
import nn.attain_v6_tiny as attain
from utils.fgsm_attack import fgm
from utils.data_utils import dataset
import utils.model_utils as  model_utils
model_utils.set_flags()
#tf.app.flags.DEFINE_string('f', '', 'kern


# In[3]:


data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)


# ## train

# In[ ]:


lr = 5e-4
valid_frequency = 1
stop_threshold = 0.98
stop_count = 5
only_test = False

#adv_path_prefixes = ["fgsm_t8"]


# In[ ]:


## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    y_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
    is_training = tf.placeholder(tf.bool, ())
    #
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
    #
    atk_model = attain.ATTAIN(x_holder, y_holder, low_bound_holder, up_bound_holder, 
                        epsilon_holder, is_training, targeted_label=tgt_label_holder)

    data_adv_ain = atk_model.prediction
    clean_acc = atk_model._target_accuracy
    adv_acc = atk_model._target_adv_accuracy
    # model
    with tf.variable_scope("ens_adv_train_target"):
        model = resnet.resnet18()
        # loss
        logits, preds = model.prediction(x_holder, use_summary=False, is_training=is_training)
        loss = model.loss(preds, y_holder)
        train_op = model.optimization(lr, loss, "ens_adv_train_target")
        acc = model.accuracy(preds, y_holder)
    #
    data_adv_ifgsm = fgm(atk_model._target.prediction, x_holder, tgt_label_holder, eps=FLAGS.EPSILON, iters=FLAGS.FGM_ITERS, 
                targeted=True, clip_min=0., clip_max=1.)

#config=tf.ConfigProto(device_count={'GPU': 0})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    model.tf_load(sess, "./models/ens_adv_train", name='tiny_cnn_ensemble_ifgsm.ckpt', scope="ens_adv_train_target", global_vars=False)
    # Load target classifier
    atk_model._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
    atk_model.tf_load(sess, name=FLAGS.AE_CKPT_RESTORE_NAME)
    atk_model.tf_load(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")
    # test
    total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    #total_test_batch = 2
    test_loss = 0; test_acc = 0
    test_ain_loss = 0; test_ain_acc = 0
    test_clean_loss = 0; test_clean_acc = 0
    test_fgsm_loss = 0; test_fgsm_acc = 0
    
    targeted_label = model_utils._one_hot_encode([int(FLAGS.TARGETED_LABEL)]*FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES)
    for test_idx in range(total_test_batch):
        batch_clean_xs, batch_clean_ys, clean_path = data.next_test_batch(FLAGS.BATCH_SIZE, True)
        feed_dict = {
            x_holder: batch_clean_xs, 
            y_holder: batch_clean_ys,
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
            
        
        #start_t = time.time()
        batch_ain_xs = sess.run(fetches=data_adv_ain, feed_dict=feed_dict)
        #bt_t = time.time() - start_t
        #import pdb; pdb.set_trace()
        #
        feed_dict = {
            x_holder: batch_clean_xs,
            tgt_label_holder: targeted_label,
            is_training: True
        }
        batch_fgsm_xs = sess.run(fetches=data_adv_ifgsm, feed_dict=feed_dict)
        
        batch_xs = np.concatenate([batch_clean_xs, batch_ain_xs, batch_fgsm_xs], 0)
        batch_ys = np.concatenate([batch_clean_ys, batch_clean_ys, batch_clean_ys], 0)
        batch_xs, batch_ys = data._shuffle_multi([batch_xs, batch_ys])
        # for clean
        feed_dict = {
            x_holder: batch_clean_xs,
            y_holder: batch_clean_ys,
            is_training: True
        }
        fetches = [loss, acc]
        batch_clean_loss, batch_clean_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_clean_loss += batch_clean_loss
        test_clean_acc += batch_clean_acc

        # for ain
        feed_dict = {
            x_holder: batch_ain_xs,
            y_holder: batch_clean_ys,
            is_training: True
        }
        fetches = [loss, acc]
        batch_ain_loss, batch_ain_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_ain_loss += batch_ain_loss
        test_ain_acc += batch_ain_acc

        # for fgsm
        #
        fetches = [loss, acc]
        feed_dict = {
            x_holder: batch_fgsm_xs,
            y_holder: batch_clean_ys,
            is_training: True
        }
        batch_fgsm_loss, batch_fgsm_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_fgsm_loss += batch_fgsm_loss
        test_fgsm_acc += batch_fgsm_acc


        # for total
        feed_dict = {
            x_holder: batch_xs,
            y_holder: batch_ys,
            is_training: True
        }
        fetches = [loss, acc]
        batch_loss, batch_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_loss += batch_loss
        test_acc += batch_acc

        

    test_clean_loss /= total_test_batch
    test_clean_acc /= total_test_batch
    test_ain_loss /= total_test_batch
    test_ain_acc /= total_test_batch
    test_fgsm_loss /= total_test_batch
    test_fgsm_acc /= total_test_batch
    test_loss /= total_test_batch
    test_acc /= total_test_batch
    print("Test loss: {:4f}, test acc: {:4f}".format(test_loss, test_acc))
    print("Test clean loss: {:4f}, test clean acc: {:4f}".format(test_clean_loss, test_clean_acc))
    print("Test ain loss: {:4f}, test ain acc: {:4f}".format(test_ain_loss, test_ain_acc))
    print("Test fgsm loss: {:4f}, test fgsm acc: {:4f}".format(test_fgsm_loss, test_fgsm_acc))

    with open("Ensemble_test_tiny_tgt_ifgsm.txt", "a+") as file: 
        file.write("Test loss: {:4f}, test acc: {:4f}\n".format(test_loss, test_acc))
        file.write("Test clean loss: {:4f}, test clean acc: {:4f}\n".format(test_clean_loss, test_clean_acc))
        file.write("Test ain loss: {:4f}, test ain acc: {:4f}\n".format(test_ain_loss, test_ain_acc))
        file.write("Test fgsm loss: {:4f}, test fgsm acc: {:4f}\n".format(test_fgsm_loss, test_fgsm_acc))
        


# In[ ]:





# In[ ]:




