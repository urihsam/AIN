#!/bin/bash
python3 ./code/run_aan.py --train --local --NUM_EPOCHS 1 --BATCH_SIZE 128 --EVAL_FREQUENCY 1\
    --AE_TYPE TRAD\
    --REG_SCALE 1e-3\
    --BETA_X_TRUE 5\
    --MAX_BETA_X_TRUE 10\
    --BETA_X_FAKE 5\
    --MAX_BETA_X_FAKE 10\
    --MODIFY_KAPPA_THRESHOLD 8\
    --KAPPA_FOR_TRANS 4\
    --KAPPA_FOR_CLEAN 4\
    --BETA_Y_TRANS 50\
    --BETA_Y_FAKE 0\
    --BETA_Y_CLEAN 50\
    --PIXEL_BOUND 0.01\
    --EPSILON 0.0025\
    --FGM_ITERS 2
