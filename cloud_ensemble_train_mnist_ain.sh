#!/bin/bash
python3 ./code/ensemble_adv_train_mnist_ain.py --NUM_EPOCHS 100 --BATCH_SIZE 100\
    --DATA_DIR ../mnist-new\
    --IMAGE_ROWS 28 --IMAGE_COLS 28 --NUM_CHANNELS 1\
    --NUM_CLASSES 10\
    --EPSILON 4e-3\
    --FGM_ITERS 50