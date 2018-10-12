#!/bin/bash
python3 ./code/run_aan.py --train --local --NUM_EPOCHS 1 --BATCH_SIZE 128 --EVAL_FREQUENCY 1\
    --AE_TYPE TRAD\
    --REG_SCALE 0.01\
    --BETA_X_TRUE 0.2\
    --MAX_BETA_X_TRUE 2\
    --BETA_X_FAKE 0.1\
    --MAX_BETA_X_FAKE 1\
    --MODIFY_KAPPA_THRESHOLD 5\
    --KAPPA_FOR_TRANS 8\
    --KAPPA_FOR_CLEAN 8\
    --BETA_Y_TRANS 100\
    --BETA_Y_FAKE 0\
    --BETA_Y_CLEAN 80\
    --PIXEL_BOUND 8\
    --EPSILON 0.5\
    --FGM_ITERS 2
