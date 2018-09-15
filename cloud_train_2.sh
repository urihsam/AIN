#!/bin/bash
python3 ./code/run_aan.py --train --NUM_EPOCHS 200 --BATCH_SIZE 128 --EVAL_FREQUENCY 25\
    --BETA_X 0.15\
    --BETA_Y 50\
    --PIXEL_THRESHOLD 8\
    --EPSILON 128 