#!/bin/bash
python3 ./code/run_aan.py --train --local --NUM_EPOCHS 1 --BATCH_SIZE 128 --EVAL_FREQUENCY 1\
    --BETA_X 0.1\
    --BETA_Y 50\
    --PIXEL_BOUND 16\
    --EPSILON 128 