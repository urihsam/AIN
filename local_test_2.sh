#!/bin/bash
python ./code/run_aan.py --local --NUM_EPOCHS 1 --BATCH_SIZE 64 --EVAL_FREQUENCY 1\
    --BETA_X 1\
    --BETA_Y_LEAST 0\
    --BETA_Y_FAKE 80\
    --BETA_Y_CLEAN 1\
    --EPSILON 64