#!/bin/bash
python ./code/run_aan.py --local --NUM_EPOCHS 1 --BATCH_SIZE 64 --EVAL_FREQUENCY 1\
    --AE_TYPE TRAD\
    --REG_SCALE 0.01\
    --BETA_X_TRUE 0.25\
    --MAX_BETA_X_TRUE 2\
    --BETA_X_FAKE 0.25\
    --MAX_BETA_X_FAKE 1\
    --MODIFY_KAPPA_THRESHOLD 8\
    --KAPPA_FOR_TRANS 8\
    --KAPPA_FOR_CLEAN 8\
    --BETA_Y_TRANS 100\
    --BETA_Y_FAKE 0\
    --BETA_Y_CLEAN 100\
    --PIXEL_BOUND 0.01\
    --EPSILON 0.05\
    --FGM_ITERS 2