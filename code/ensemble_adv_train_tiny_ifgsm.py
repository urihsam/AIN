
# coding: utf-8

# In[1]:


import sys
import os
#PATH = "../"
import time

# In[2]:


#sys.path.append(os.path.abspath(PATH))
from dependency import *
import nn.resnet as resnet
from PIL import Image
import nn.attain_v6_tiny as attain
from utils.data_utils import dataset
import utils.model_utils as  model_utils
model_utils.set_flags()
#tf.app.flags.DEFINE_string('f', '', 'kernel')


# In[3]:

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_INDEX
data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED, adv_path_prefix=FLAGS.ADV_PATH_PREFIX)


# ## train

# In[ ]:


lr = 1e-5
valid_frequency = 1
stop_threshold = 0.98
stop_count = 5
only_test = False

#adv_path_prefixes = ["fgsm_t0", "fgsm_t1", "fgsm_t2", "fgsm_t3", "fgsm_t4", "fgsm_t5", "fgsm_t6", "fgsm_t7", "fgsm_t8", "fgsm_t9"]
adv_path_prefixes = [FLAGS.ADV_PATH_PREFIX]


# In[ ]:


## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
    y_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
    is_training = tf.placeholder(tf.bool, ())
    # model
    with tf.variable_scope("ens_adv_train_target"):
        model = resnet.resnet18()
        # loss
        logits, preds = model.prediction(x_holder, use_summary=False, is_training=is_training)
        loss = model.loss(preds, y_holder)
        train_op = model.optimization(lr, loss, r'ens_adv_train_target')
        acc = model.accuracy(preds, y_holder)

#config=tf.ConfigProto(device_count={'GPU': 0})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    # init model
    #import pdb; pdb.set_trace()
    model.tf_load(sess, FLAGS.ENS_RESNET18_PATH, 'model.ckpt-5865', scope="ens_adv_train_target", global_vars=False)
    #
    total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
    total_valid_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
    total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    
    if only_test == False:
        # training, validation and test
        count = 0
        for epoch in range(FLAGS.NUM_EPOCHS):
            print("Epoch {}: ".format(epoch))
            #start_t = time.time()
            for train_idx in range(total_train_batch):
                batch_clean_xs, batch_clean_ys, batch_adv_xs = data.next_train_batch(FLAGS.BATCH_SIZE, False)
                batch_xs = np.concatenate([batch_clean_xs, batch_adv_xs], 0)
                batch_ys = np.concatenate([batch_clean_ys, batch_clean_ys], 0)
                batch_xs, batch_ys = data._shuffle_multi([batch_xs, batch_ys])
                feed_dict = {
                    x_holder: batch_xs,
                    y_holder: batch_ys,
                    is_training: True
                }
                fetches = [train_op, loss, acc]
                _, train_loss, train_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                if train_idx % 10 == 9:
                    print("Training loss: {:4f}, training acc: {:4f}".format(train_loss, train_acc))
            #ep_t = time.time() - start_t
            #import pdb; pdb.set_trace()
            if epoch % valid_frequency == valid_frequency-1:
                valid_loss = 0; valid_acc = 0
                valid_adv_loss = 0; valid_adv_acc = 0
                valid_clean_loss = 0; valid_clean_acc = 0
                for valid_idx in range(total_valid_batch):
                    batch_clean_xs, batch_clean_ys, batch_adv_xs = data.next_valid_batch(FLAGS.BATCH_SIZE, False)
                    #
                    batch_xs = np.concatenate([batch_clean_xs, batch_adv_xs], 0)
                    batch_ys = np.concatenate([batch_clean_ys, batch_clean_ys], 0)
                    batch_xs, batch_ys = data._shuffle_multi([batch_xs, batch_ys])
                    # for clean
                    feed_dict = {
                        x_holder: batch_clean_xs,
                        y_holder: batch_clean_ys,
                        is_training: True
                    }
                    fetches = [loss, acc]
                    batch_clean_loss, batch_clean_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                    valid_clean_loss += batch_clean_loss
                    valid_clean_acc += batch_clean_acc
                    
                    # for adv
                    feed_dict = {
                        x_holder: batch_adv_xs,
                        y_holder: batch_clean_ys,
                        is_training: True
                    }
                    fetches = [loss, acc]
                    batch_adv_loss, batch_adv_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                    valid_adv_loss += batch_adv_loss
                    valid_adv_acc += batch_adv_acc
                    
                    # for total
                    feed_dict = {
                        x_holder: batch_xs,
                        y_holder: batch_ys,
                        is_training: True
                    }
                    fetches = [loss, acc]
                    batch_loss, batch_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                    valid_loss += batch_loss
                    valid_acc += batch_acc
                valid_clean_loss /= total_valid_batch
                valid_clean_acc /= total_valid_batch
                valid_adv_loss /= total_valid_batch
                valid_adv_acc /= total_valid_batch
                valid_loss /= total_valid_batch
                valid_acc /= total_valid_batch
                print("Validation loss: {:4f}, validation acc: {:4f}".format(valid_loss, valid_acc))
                print("Validation clean loss: {:4f}, validation clean acc: {:4f}".format(valid_clean_loss, valid_clean_acc))
                print("Validation adv loss: {:4f}, validation adv acc: {:4f}".format(valid_adv_loss, valid_adv_acc))
                '''
                if valid_acc > 0.98:
                    if count > stop_count: break
                    else: count += 1
                else: count = 0
                '''
    # test
    test_loss = 0; test_acc = 0
    test_adv_loss = 0; test_adv_acc = 0
    test_clean_loss = 0; test_clean_acc = 0
    for test_idx in range(total_test_batch):
        batch_clean_xs, batch_clean_ys, batch_adv_xs = data.next_test_batch(FLAGS.BATCH_SIZE, False)
        #
        batch_xs = np.concatenate([batch_clean_xs, batch_adv_xs], 0)
        batch_ys = np.concatenate([batch_clean_ys, batch_clean_ys], 0)
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

        # for adv
        feed_dict = {
            x_holder: batch_adv_xs,
            y_holder: batch_clean_ys,
            is_training: True
        }
        fetches = [loss, acc]
        batch_adv_loss, batch_adv_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_adv_loss += batch_adv_loss
        test_adv_acc += batch_adv_acc

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
    test_adv_loss /= total_test_batch
    test_adv_acc /= total_test_batch
    test_loss /= total_test_batch
    test_acc /= total_test_batch
    print("Test loss: {:4f}, test acc: {:4f}".format(test_loss, test_acc))
    print("Test clean loss: {:4f}, test clean acc: {:4f}".format(test_clean_loss, test_clean_acc))
    print("Test adv loss: {:4f}, test adv acc: {:4f}".format(test_adv_loss, test_adv_acc))

    model.tf_save(sess, "./models/ens_adv_train", name='tiny_cnn_ensemble_ifgsm.ckpt', scope="ens_adv_train_target")
        


# In[ ]:





# In[ ]:




