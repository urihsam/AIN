
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
import utils.model_utils_mnist as  model_utils
model_utils.set_flags()
#tf.app.flags.DEFINE_string('f', '', 'kernel')


# In[3]:
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_INDEX

data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)


# ## train

# In[ ]:


lr = 5e-4
valid_frequency = 1
stop_threshold = 0.97
stop_count = 5
only_test = False

adv_path_prefixes = ["AIN_t8"]


# In[ ]:


## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_holder = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool, ())
    # model
    with tf.variable_scope("target"):
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

#config=tf.ConfigProto(device_count={'GPU': 0})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
    total_valid_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
    total_test_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    
    # training, validation and test
    if only_test == False:
        count = 0
        for epoch in range(FLAGS.NUM_EPOCHS):
            print("Epoch {}: ".format(epoch))
            for train_idx in range(total_train_batch):
                batch_clean_xs, batch_clean_ys, clean_path = data.next_train_batch(FLAGS.BATCH_SIZE, True)
                batch_adv_xs, batch_adv_ys = data._load_batch_adv_images_from_adv_path_prefixes(FLAGS.BATCH_SIZE, clean_path, batch_clean_ys, adv_path_prefixes)
                batch_xs = np.concatenate([batch_clean_xs, batch_adv_xs], 0)
                batch_ys = np.concatenate([batch_clean_ys, batch_adv_ys], 0)
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
                
            if epoch % valid_frequency == valid_frequency-1:
                valid_loss = 0; valid_acc = 0
                valid_adv_loss = 0; valid_adv_acc = 0
                valid_clean_loss = 0; valid_clean_acc = 0
                for valid_idx in range(total_valid_batch):
                    batch_clean_xs, batch_clean_ys, clean_path = data.next_valid_batch(FLAGS.BATCH_SIZE, True)
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
                    valid_clean_loss += batch_clean_loss
                    valid_clean_acc += batch_clean_acc
                    
                    # for adv
                    feed_dict = {
                        x_holder: batch_adv_xs,
                        y_holder: batch_adv_ys,
                        is_training: False
                    }
                    fetches = [loss, acc]
                    batch_adv_loss, batch_adv_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                    valid_adv_loss += batch_adv_loss
                    valid_adv_acc += batch_adv_acc
                    
                    # for total
                    feed_dict = {
                        x_holder: batch_xs,
                        y_holder: batch_ys,
                        is_training: False
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
                if valid_acc > 0.98:
                    if count > stop_count: break
                    else: count += 1
                else: count = 0
    # test
    test_loss = 0; test_acc = 0
    test_adv_loss = 0; test_adv_acc = 0
    test_clean_loss = 0; test_clean_acc = 0
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
    test_clean_loss /= total_test_batch
    test_clean_acc /= total_test_batch
    test_adv_loss /= total_test_batch
    test_adv_acc /= total_test_batch
    test_loss /= total_test_batch
    test_acc /= total_test_batch
    print("Test loss: {:4f}, test acc: {:4f}".format(test_loss, test_acc))
    print("Test clean loss: {:4f}, test clean acc: {:4f}".format(test_clean_loss, test_clean_acc))
    print("Test adv loss: {:4f}, test adv acc: {:4f}".format(test_adv_loss, test_adv_acc))

    model.tf_save(sess, "./models", "target", name='mnist_cnn_ensemble_ain.ckpt')
        


# In[ ]:





# In[ ]:




