#!/bin/bash
python3 ./code/CW_attack_mnist.py --NUM_EPOCHS 1 --BATCH_SIZE 1\
    --DATA_DIR ../mnist\
    --IMAGE_ROWS 28 --IMAGE_COLS 28 --NUM_CHANNELS 1\
    --NUM_CLASSES 10