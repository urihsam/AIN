#!/bin/bash
python3 ./code/CW_attack_tiny.py --NUM_EPOCHS 1 --BATCH_SIZE 1\
    --DATA_DIR ../tiny-imagenet-200\
    --IMAGE_ROWS 64 --IMAGE_COLS 64 --NUM_CHANNELS 3\
    --NUM_CLASSES 200