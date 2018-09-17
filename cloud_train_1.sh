#!/bin/bash
python3 ./code/run_aan.py --train --NUM_EPOCHS 200 --BATCH_SIZE 128 --EVAL_FREQUENCY 25\
    --BETA_X 0.1\
    --BETA_Y 50\
    --PIXEL_BOUND 16\
    --BOUND_DECAY_EPOCHS 4\
    --EPSILON_DECAY_EPOCHS 4\
    --EPSILON 128 