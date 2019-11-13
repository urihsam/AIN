
# coding: utf-8

# In[1]:

import sys
import os


# In[2]:

from dependency import *
from PIL import Image
import utils.model_utils_imagenet as model_utils
from utils.data_utils_imagenet import dataset
import nn.imagenet_keras as resnet
#import nn.imagenet_resnet50 as resnet

model_utils.set_flags()

# In[3]:
DATA_DIR = "../imagenet2012_rz_224/imagenet2012_rz"
#DATA_DIR = "../imagenet2011_rz"
data = dataset(DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)

weight_path = "./models/target_classifier/resnet152/resnet152_weights_tf.h5"

# ## train

# In[4]:

BATCH_SIZE = 20
NUM_EPOCHS = 50
IMAGE_ROWS = 224
IMAGE_COLS = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1000
lr = 1e-5
valid_frequency = 1
stop_threshold = 0.8
stop_count = 5


# In[5]:

## tf.reset_default_graph()
g = tf.get_default_graph()
# attack_target = 8
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [None, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS])
    y_holder = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    is_training = tf.placeholder(tf.bool, ())
    # model
    with tf.variable_scope("target"):
        model = resnet.resnet50(weight_path)
        #model = resnet.resnet50()
        # loss
        logits, preds = model.evaluate(x_holder, is_training)
    loss = 10*model.loss(logits, y_holder, loss_type="categ")
    train_op = model.optimization(lr, loss)
    acc = model.accuracy(preds, y_holder)
    #import pdb; pdb.set_trace()


with tf.Session(graph=g) as sess:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    model.load_weight()
    #model.tf_load(sess, "./models/target_classifier/resnet_v2_50/resnet_v2_50_2017", "target", name='resnet_v2_50.ckpt')
    total_train_batch = int(data.train_size/BATCH_SIZE)
    total_valid_batch = int(data.valid_size/BATCH_SIZE)
    total_test_batch = int(data.test_size/BATCH_SIZE)
    
    # training, validation and test
    count = 0
    for epoch in range(NUM_EPOCHS):
        print("Epoch {}: ".format(epoch))
        for train_idx in range(total_train_batch):
            batch_xs, batch_ys = data.next_train_batch(BATCH_SIZE)
            feed_dict = {
                x_holder: batch_xs,
                y_holder: batch_ys,
                is_training: True
            }
            #import pdb; pdb.set_trace()
            res = sess.run([logits, preds], feed_dict=feed_dict)

            fetches = [train_op, loss, acc]
            _, train_loss, train_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
            if train_idx % 2500 == 2499:
                print("Training loss: {:4f}, training acc: {:4f}".format(train_loss, train_acc))
            
        if epoch % valid_frequency == valid_frequency-1:
            valid_loss = 0; valid_acc = 0
            for valid_idx in range(total_valid_batch):
                batch_xs, batch_ys = data.next_valid_batch(BATCH_SIZE)
                feed_dict = {
                    x_holder: batch_xs,
                    y_holder: batch_ys,
                    is_training: False
                }
                fetches = [loss, acc]
                batch_loss, batch_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                valid_loss += batch_loss
                valid_acc += batch_acc
            valid_loss /= total_valid_batch
            valid_acc /= total_valid_batch
            print("Validation loss: {:4f}, validation acc: {:4f}".format(valid_loss, valid_acc))
            model.tf_save(sess, FLAGS.RESNET50_PATH, "target", "resnetv2_50_keras_1113.ckpt")
            if valid_acc > stop_threshold:
                if count > stop_count: break;
                else: count += 1
            else: count = 0
    # test
    test_loss = 0; test_acc = 0
    for test_idx in range(total_test_batch):
        batch_xs, batch_ys = data.next_test_batch(BATCH_SIZE)
        feed_dict = {
            x_holder: batch_xs,
            y_holder: batch_ys,
            is_training: False
        }
        fetches = [loss, acc]
        batch_loss, batch_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_loss += batch_loss
        test_acc += batch_acc
    test_loss /= total_test_batch
    test_acc /= total_test_batch
    print("Test loss: {:4f}, test acc: {:4f}".format(test_loss, test_acc))
    model.tf_save(sess, FLAGS.RESNET50_PATH, "target", "resnetv2_50_keras_1113.ckpt")
        


# In[ ]:



