
# coding: utf-8

# In[1]:


import sys
import os
#PATH = "../"


# In[2]:


#sys.path.append(os.path.abspath(PATH))
from dependency import *
from nn.mnist_classifier import MNISTCNN
from PIL import Image
from utils.data_utils_mnist import dataset
from utils.fgsm_attack import fgm
import utils.model_utils_mnist as  model_utils
model_utils.set_flags()
#tf.app.flags.DEFINE_string('f', '', 'kernel')


# In[3]:


data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)


# ## train

# In[ ]:


lr = 5e-4
valid_frequency = 1
stop_threshold = 0.98
stop_count = 5
only_test = False

#adv_path_prefixes = ["fgsm_t0", "fgsm_t1", "fgsm_t2", "fgsm_t3", "fgsm_t4", "fgsm_t5", "fgsm_t6", "fgsm_t7", "fgsm_t8", "fgsm_t9"]
adv_path_prefixes = ["AIN_t8"]


# In[ ]:


## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_holder = tf.placeholder(tf.float32, [None, 10])
    tgt_y_holder = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, ())
    # model
    with tf.variable_scope("target") as scope:
        model = MNISTCNN(conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
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
                         img_channel=1
                        )
    # loss
    logits, preds = model.evaluate(x_holder, is_training)
    loss = model.loss(logits, y_holder, loss_type="mse")
    train_op = model.optimization(lr, loss)
    acc = model.accuracy(preds, y_holder)
    #
    data_adv = fgm(model.prediction, x_holder, tgt_y_holder, eps=FLAGS.EPSILON, iters=FLAGS.FGM_ITERS, 
                targeted=True, clip_min=0., clip_max=1.)

#config=tf.ConfigProto(device_count={'GPU': 0})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    model.tf_load(sess, "./models", "target", name='mnist_cnn_ensemble_ifgsm.ckpt')
    #total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    total_test_batch = 2
    # test
    test_loss = 0; test_acc = 0
    test_adv_loss = 0; test_adv_acc = 0
    test_clean_loss = 0; test_clean_acc = 0
    test_fgsm_loss = 0; test_fgsm_acc = 0
    for test_idx in range(total_test_batch):
        batch_clean_xs, batch_clean_ys, clean_path = data.next_test_batch(FLAGS.BATCH_SIZE, True)
        batch_adv_xs, batch_adv_ys = data._load_batch_adv_images_from_adv_path_prefixes(FLAGS.BATCH_SIZE, clean_path, batch_clean_ys, adv_path_prefixes)
        batch_adv_xs, batch_adv_ys = data._shuffle_multi([batch_adv_xs, batch_adv_ys])
        #
        batch_xs = np.concatenate([batch_clean_xs, batch_adv_xs], 0)
        batch_ys = np.concatenate([batch_clean_ys, batch_adv_ys], 0)
        batch_xs, batch_ys = data._shuffle_multi([batch_xs, batch_ys])
        # for clean
        feed_dict = {
            x_holder: batch_clean_xs,
            y_holder: batch_clean_ys,
            is_training: False
        }
        fetches = [loss, acc]
        batch_clean_loss, batch_clean_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_clean_loss += batch_clean_loss
        test_clean_acc += batch_clean_acc

        # for adv
        feed_dict = {
            x_holder: batch_adv_xs,
            y_holder: batch_adv_ys,
            is_training: False
        }
        fetches = [loss, acc]
        batch_adv_loss, batch_adv_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_adv_loss += batch_adv_loss
        test_adv_acc += batch_adv_acc

        # for total
        feed_dict = {
            x_holder: batch_xs,
            y_holder: batch_ys,
            is_training: False
        }
        fetches = [loss, acc]
        batch_loss, batch_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_loss += batch_loss
        test_acc += batch_acc

        # for fgsm
        batch_tgt_ys = np.asarray(model_utils._one_hot_encode([8]*FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES))
        feed_dict = {
            x_holder: batch_clean_xs,
            tgt_y_holder: batch_tgt_ys,
            is_training: False
        }
        batch_fgsm_xs = sess.run(fetches=data_adv, feed_dict=feed_dict)
        #
        fetches = [loss, acc]
        feed_dict = {
            x_holder: batch_fgsm_xs,
            y_holder: batch_clean_ys,
            is_training: False
        }
        batch_fgsm_loss, batch_fgsm_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_fgsm_loss += batch_fgsm_loss
        test_fgsm_acc += batch_fgsm_acc

    test_clean_loss /= total_test_batch
    test_clean_acc /= total_test_batch
    test_adv_loss /= total_test_batch
    test_adv_acc /= total_test_batch
    test_fgsm_loss /= total_test_batch
    test_fgsm_acc /= total_test_batch
    test_loss /= total_test_batch
    test_acc /= total_test_batch
    print("Test loss: {:4f}, test acc: {:4f}".format(test_loss, test_acc))
    print("Test clean loss: {:4f}, test clean acc: {:4f}".format(test_clean_loss, test_clean_acc))
    print("Test adv loss: {:4f}, test adv acc: {:4f}".format(test_adv_loss, test_adv_acc))
    print("Test fgsm loss: {:4f}, test fgsm acc: {:4f}".format(test_fgsm_loss, test_fgsm_acc))

    with open("Ensemble_test_mnist_ifgsm.txt", "a+") as file: 
        file.write("Test loss: {:4f}, test acc: {:4f}".format(test_loss, test_acc))
        file.write("Test clean loss: {:4f}, test clean acc: {:4f}".format(test_clean_loss, test_clean_acc))
        file.write("Test adv loss: {:4f}, test adv acc: {:4f}".format(test_adv_loss, test_adv_acc))
        file.write("Test fgsm loss: {:4f}, test fgsm acc: {:4f}".format(test_fgsm_loss, test_fgsm_acc))
        


# In[ ]:





# In[ ]:




