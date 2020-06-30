#!/bin/bash
python3 ./code/fgsm_attack_tiny_diversity.py --NUM_EPOCHS 1 --BATCH_SIZE 500\
    --DATA_DIR ../tiny-imagenet-200\
    --IS_TARGETED_ATTACK=True --ADV_PATH_PREFIX fgsm_t8\
    --IMAGE_ROWS 64 --IMAGE_COLS 64 --NUM_CHANNELS 3\
    --NUM_CLASSES 200\
    --EPSILON 2e-4\
    --FGM_ITERS 50