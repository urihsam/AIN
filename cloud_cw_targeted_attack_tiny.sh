#!/bin/bash
python3 ./code/CW_targeted_attack_tiny.py --NUM_EPOCHS 1 --BATCH_SIZE 1\
    --GPU_INDEX 0\
    --DATA_DIR ../tiny-imagenet-200 --TARGETED_LABEL 8\
    --IMAGE_ROWS 64 --IMAGE_COLS 64 --NUM_CHANNELS 3\
    --NUM_CLASSES 200