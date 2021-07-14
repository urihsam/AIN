#!/bin/bash
python ./code/AIN_targeted_attack_mnist.py --train --NUM_EPOCHS 1 --BATCH_SIZE 100 --EVAL_FREQUENCY 400 --VALID_FREQUENCY 4\
    --GPU_INDEX 1\
    --IS_TARGETED_ATTACK=True --TARGETED_LABEL 8\
    --DATA_DIR ../mnist-new\
    --AE_TYPE ATTAE --AE_PATH ./models/AE_TGT\
    --ATT_TYPE TRAD\
    --load_AE=True --AE_CKPT_RESTORE_NAME deep_cae.Linf0.204095.Lx3.906427.acc0.910500.ckpt\
    --INIT_MAX_VALID_ACC 0.91\
    --train_label=False \
    --ADD_RANDOM=True \
    --USE_LABEL_MASK=True \
    --NUM_PRE_EPOCHS 15 --PRE_EVAL_FREQUENCY 1\
    --NUM_ACCUM_ITERS 1\
    --GAMMA_R 1e-3 --GAMMA_L 1.0\
    --REG_SCALE 1e-1\
    --OPT_TYPE ADAM --LEARNING_RATE 5e-5 --LEARNING_DECAY_RATE 0.99 --LEARNING_DECAY_STEPS 250\
    --IS_GRAD_CLIPPING --GRAD_CLIPPING_NORM 1.0\
    --LOSS_MODE_TRANS C_W2 --LOSS_MODE_FAKE MIX --LOSS_MODE_CLEAN C_W2\
    --BETA_X_TRUE 5.0 --BETA_X_TRUE_CHANGE_RATE 1.02\
    --BETA_X_FAKE 5.0 --BETA_X_FAKE_CHANGE_RATE 1.02\
    --LOSS_Y_LOW_BOUND_T -50.0 --LOSS_Y_UP_BOUND_T 50.0\
    --LOSS_Y_LOW_BOUND_F -50.0 --LOSS_Y_UP_BOUND_F 50.0\
    --LOSS_Y_LOW_BOUND_C -50.0 --LOSS_Y_UP_BOUND_C 50.0\
    --BETA_Y_TRANS 10.0 --BETA_Y_TRANS_CHANGE_RATE 1.001\
    --BETA_Y_FAKE 10.0 --BETA_Y_FAKE_CHANGE_RATE 1.001\
    --BETA_Y_FAKE2 10.0 --BETA_Y_FAKE2_CHANGE_RATE 1.001\
    --BETA_Y_CLEAN 1.0 --BETA_Y_CLEAN_CHANGE_RATE 1.001\
    --BOUND_CHANGE_TYPE EXP --PIXEL_BOUND 0.204 --BOUND_CHANGE_RATE -0.02 --BOUND_CHANGE_EPOCHS 8\
    --MIN_BOUND 0.001 --MAX_BOUND 1.0\
    --ROLL_BACK_THRESHOLD 4\
    --ABS_DIFF_THRESHOLD 1e-4\
    --ADAPTIVE_UP_THRESHOLD 1e-4 --ADAPTIVE_LOW_THRESHOLD 2e-5\
    --ADAPTIVE_BOUND_INC_RATE 0.6 --ADAPTIVE_BOUND_DEC_RATE 1.4\
    --ENC_NORM LAYER --DEC_NORM LAYER\
    --CENTRAL_CHANNEL_SIZE 3\
    --NUM_ENC_RES_BLOCK 4 --ENC_RES_BLOCK_SIZE 1\
    --NUM_DEC_RES_BLOCK 4 --DEC_RES_BLOCK_SIZE 1\
    --EPSILON 4e-3\
    --FGM_ITERS 50