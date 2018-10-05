#!/bin/bash
python3 ./code/run_aan.py --train --local --NUM_EPOCHS 1 --BATCH_SIZE 128 --EVAL_FREQUENCY 1\
    --AE_TYPE VARI\
    --EARLY_STOPPING_THRESHOLD 50\
    --BETA_X_TRUE 0.25\
    --BETA_X_FAKE 0.25\
    --BETA_Y_TRANS 100\
    --BETA_Y_FAKE 0\
    --BETA_Y_CLEAN 100\
    --PIXEL_BOUND 16\
    --EPSILON 0.5\
    --FGM_ITERS 2