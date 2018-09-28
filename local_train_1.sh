#!/bin/bash
python3 ./code/run_aan.py --train --local --NUM_EPOCHS 1 --BATCH_SIZE 64 --EVAL_FREQUENCY 1\
    --AE_TYPE TRAD\
    --EARLY_STOPPING_THRESHOLD 10\
    --BETA_X_TRUE 0.05\
    --BETA_X_FAKE 0.05\
    --BETA_Y_TRANS 20\
    --BETA_Y_FAKE 50\
    --BETA_Y_CLEAN 1\
    --PIXEL_BOUND 64\
    --EPSILON 1\
    --FGM_ITERS 2