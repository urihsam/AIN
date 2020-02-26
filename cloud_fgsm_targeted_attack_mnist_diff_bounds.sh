#!/bin/bash
python3 ./code/fgsm_targeted_attack_mnist_diff_bounds.py --NUM_EPOCHS 1 --BATCH_SIZE 500\
    --DATA_DIR ../mnist\
    --TARGETED_LABEL 8\
    --IMAGE_ROWS 28 --IMAGE_COLS 28 --NUM_CHANNELS 1\
    --NUM_CLASSES 10\
    --EPSILON 4e-3\
    --FGM_ITERS 50