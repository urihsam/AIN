
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
from utils.data_utils_mnist import dataset

model_utils.set_flags()

data = dataset(FLAGS.DATA_DIR, split_ratio=1.0, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED, 
    adv_path_prefix=FLAGS.ADV_PATH_PREFIX)
diff = []
ys = []
for i in range(data.test_size//FLAGS.BATCH_SIZE):
    batch_xs, batch_ys, batch_adv = data.next_test_batch(FLAGS.BATCH_SIZE, with_path=False)
    diff.append(batch_adv - batch_xs)
    ys.append(batch_ys)
diff = np.concatenate(diff, 0)
ys = np.concatenate(ys, 0)


if FLAGS.IS_TARGETED_ATTACK:
    np.save("fgsm_tgt_ys_diversity.npy", ys)
    np.save("fgsm_tgt_diversity.npy", diff)
else:
    np.save("fgsm_untgt_ys_diversity.npy", ys)
    np.save("fgsm_untgt_diversity.npy", diff)


    

