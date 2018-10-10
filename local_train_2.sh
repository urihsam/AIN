#!/bin/bash
python3 ./code/run_aan.py --train --local --NUM_EPOCHS 1 --BATCH_SIZE 128 --EVAL_FREQUENCY 1\
    --AE_TYPE TRAD\
    --EARLY_STOPPING_THRESHOLD 20\
    --REG_SCALE 0.001\
    --BETA_X_TRUE 0.05\
    --BETA_X_FAKE 0.025\
    --BETA_Y_TRANS 20\
    --BETA_Y_FAKE 0\
    --BETA_Y_CLEAN 10\
    --PIXEL_BOUND 8\
    --EPSILON 0.5\
    --FGM_ITERS 2
