#!/bin/bash
python3 ./code/run_aan.py --train --NUM_EPOCHS 500 --BATCH_SIZE 128 --EVAL_FREQUENCY 25\
    --AE_TYPE VARI\
    --GAMMA_V 1e-2\
    --REG_SCALE 0.01\
    --BETA_X_TRUE 0.5\
    --BETA_X_TRUE_CHANGE_RATE 1.2\
    --BETA_X_TRUE_CHANGE_EPOCHS 5\
    --MAX_BETA_X_TRUE 1\
    --BETA_X_FAKE 0.25\
    --BETA_X_FAKE_CHANGE_RATE 1.2\
    --BETA_X_FAKE_CHANGE_EPOCHS 5\
    --MAX_BETA_X_FAKE 0.5\
    --MODIFY_KAPPA_THRESHOLD 8\
    --KAPPA_FOR_TRANS 8\
    --KAPPA_TRANS_CHANGE_RATE 1.2\
    --KAPPA_FOR_CLEAN 8\
    --KAPPA_CLEAN_CHANGE_RATE 1.2\
    --BETA_Y_TRANS 200\
    --BETA_Y_FAKE 0\
    --BETA_Y_CLEAN 200\
    --PIXEL_BOUND 8\
    --EPSILON 0.5\
    --FGM_ITERS 2