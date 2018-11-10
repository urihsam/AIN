#!/bin/bash
python3 ./code/run_aan.py --train --local --NUM_EPOCHS 1 --BATCH_SIZE 128 --EVAL_FREQUENCY 1\
    --AE_TYPE VARI\
    --GAMMA_V 1e-0\
    --REG_SCALE 1e-2\
    --BETA_X_TRUE 25\
    --MAX_BETA_X_TRUE 50\
    --BETA_X_FAKE 25\
    --MAX_BETA_X_FAKE 50\
    --MODIFY_KAPPA_THRESHOLD 8\
    --KAPPA_FOR_TRANS 6\
    --KAPPA_FOR_CLEAN 6\
    --BETA_Y_TRANS 75\
    --BETA_Y_FAKE 0\
    --BETA_Y_CLEAN 75\
    --PIXEL_BOUND 0.1\
    --MIN_BOUND 0.005\
    --EPSILON 0.0025\
    --FGM_ITERS 2         
