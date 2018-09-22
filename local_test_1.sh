#!/bin/bash
python ./code/run_aan.py --local --NUM_EPOCHS 1 --BATCH_SIZE 128 --EVAL_FREQUENCY 1\
    --BETA_X 0.1\
    --BETA_Y_LEAST 0\
    --BETA_Y_FAKE 50\
    --BETA_Y_CLEAN 1\
    --PIXEL_BOUND 64\
    --EPSILON 64 