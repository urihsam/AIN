#!/bin/bash
python3 ./code/fgsm_targeted_attack_tiny.py --NUM_EPOCHS 1 --BATCH_SIZE 1\
    --DATA_DIR ../tiny-imagenet-200\
    --TARGETED_LABEL 8\
    --IMAGE_ROWS 64 --IMAGE_COLS 64 --NUM_CHANNELS 3\
    --NUM_CLASSES 200\
    --EPSILON 2e-4\
    --FGM_ITERS 50