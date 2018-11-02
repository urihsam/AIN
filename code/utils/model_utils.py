from dependency import *
import os

def set_flags():
    flags.DEFINE_bool("train", False, "Train and save the ATN model.")
    flags.DEFINE_bool("local", False, "Run this model locally or on the cloud")
    # Path
    flags.DEFINE_string("RESNET18_PATH", "./models/target_classifier/resnet18", "Path of Resnet18")
    flags.DEFINE_string("AE_PATH", "./models/AE", "Path of AAN")
    flags.DEFINE_string("CNN_PATH", "./models/target_classifier/basic_CNN", "Path of CNN")
    flags.DEFINE_string("TRAIN_LOG_PATH", "./graphs/train", "Path of log for training")
    flags.DEFINE_string("VALID_LOG_PATH", "./graphs/valid", "Path of log for validation")
    flags.DEFINE_string("TEST_LOG_PATH", "./graphs/test", "Path of log for testing")
    flags.DEFINE_string("DATA_DIR", "../../tiny-imagenet-200", "Data dir")
    # Data description
    flags.DEFINE_integer("NUM_CLASSES", 200, "Number of classification classes")
    flags.DEFINE_integer("IMAGE_ROWS", 64, "Input row dimension")
    flags.DEFINE_integer("IMAGE_COLS", 64, "Input column dimension")
    flags.DEFINE_integer("NUM_CHANNELS", 3, "Input depth dimension")
    # Training params
    flags.DEFINE_integer("NUM_EPOCHS", 1, "Number of epochs") # 200
    flags.DEFINE_integer("BATCH_SIZE", 128, "Size of training batches")# 128
    flags.DEFINE_integer("EVAL_FREQUENCY", 1, "Frequency for evaluation") # 25
    flags.DEFINE_bool("load_AE", False, "Load AE from the last training result or not")
    flags.DEFINE_bool("early_stopping", False, "Use early stopping or not")
    flags.DEFINE_integer("EARLY_STOPPING_THRESHOLD", 10, "Early stopping threshold")
    # AE type
    flags.DEFINE_string("AE_TYPE", "TRAD", "The type of Autoencoder") # SPARSE, VARI(ATIONAL), TRAD(ITIONAL), ATTEN(TIVE)
    flags.DEFINE_bool("SPARSE", False, "The type of Autoencoder") # SPARSE, VARI(ATIONAL), TRAD(ITIONAL), ATTEN(TIVE)
    flags.DEFINE_bool("VARI", False, "The type of Autoencoder") # SPARSE, VARI(ATIONAL), TRAD(ITIONAL), ATTEN(TIVE)
    # Loss params
    flags.DEFINE_string("LOSS_MODE_TRANS", "C&W", "How to calculate loss from fake imgae") # ENTRO, C&W
    flags.DEFINE_string("LOSS_MODE_FAKE", "C&W", "How to calculate loss from fake imgae") # LOGITS, PREDS, ENTRO, C&W
    flags.DEFINE_string("LOSS_MODE_CLEAN", "C&W", "How to calculate loss from clean image") # LOGITS, PREDS, ENTRO, C&W
    ##
    flags.DEFINE_string("NORM_TYPE", "L2", "The norm type") # INF, L2, L1
    flags.DEFINE_float('REG_SCALE', 0.01, 'The scale of regularization')
    ## Gamma for rho distance
    flags.DEFINE_float("SPARSE_RHO", 10, "The sparse threshold for central states of AE")
    flags.DEFINE_float("GAMMA_S", 1e-5, "Coefficient for RHO distance") # 0.01
    ## Gamma for variational kl distance
    flags.DEFINE_float("GAMMA_V", 1e-4, "Coefficient for KL distance") # 1e-3
    ## loss x
    flags.DEFINE_string("PARTIAL_LOSS", "FULL_LOSS", "Use loss x or loss y") # FULL_LOSS, LOSS_X, LOSS_Y
    flags.DEFINE_integer("LOSS_CHANGE_FREQUENCY", 5, "The frequency of changing loss") # 5
    flags.DEFINE_float("LOSS_X_THRESHOLD", 500, "Threshold for loss x") # 500
    flags.DEFINE_float("LOSS_X_THRE_CHANGE_RATE", 0.8, "Change rate of Loss x threshold") # 0.8
    flags.DEFINE_float("LOSS_X_THRE_CHANGE_EPOCHS", 10, "Change epochs of Loss x threshold") # 10
    flags.DEFINE_float("MIN_LOSS_X_THRE", 50, "Minimum threshold for loss x") # 50
    flags.DEFINE_float("MAX_LOSS_X_THRE", 1500, "Maximum threshold for loss x") # 150
    ## Kappa: C&W loss
    flags.DEFINE_integer("MODIFY_KAPPA_THRESHOLD", 5, "Modify beta y threshold")
    flags.DEFINE_float("KAPPA_FOR_TRANS", 8, "The min logits distance") # 8
    flags.DEFINE_float("KAPPA_TRANS_CHANGE_RATE", 1.2, "The change rate of KAPPA TRANS")
    flags.DEFINE_float("KAPPA_FOR_FAKE", 8, "The min logits distance") # 8
    flags.DEFINE_float("KAPPA_FAKE_CHANGE_RATE", 1.2, "The change rate of KAPPA FAKE")
    flags.DEFINE_float("KAPPA_FOR_CLEAN", 8, "The min logits distance") # 8
    flags.DEFINE_float("KAPPA_CLEAN_CHANGE_RATE", 1.2, "The change rate of KAPPA CLEAN")
    ## Beta x true
    flags.DEFINE_float("BETA_X_TRUE", 0.1, "Coefficient for loss of X") # 1
    flags.DEFINE_float("BETA_X_TRUE_CHANGE_RATE", 1.2, "Change rate of Beta x") # 1.2
    flags.DEFINE_float("BETA_X_TRUE_CHANGE_EPOCHS", 10, "Change epochs of Beta x") # 10
    flags.DEFINE_float("MIN_BETA_X_TRUE", 0.01, "Minimum of beta x") # 0.01
    flags.DEFINE_float("MAX_BETA_X_TRUE", 1, "Maximum of beta x") # 1
    ## Beta x fake
    flags.DEFINE_float("BETA_X_FAKE", 0.1, "Coefficient for loss of X") # 1
    flags.DEFINE_float("BETA_X_FAKE_CHANGE_RATE", 1.2, "Change rate of Beta x") # 1.2
    flags.DEFINE_float("BETA_X_FAKE_CHANGE_EPOCHS", 10, "Change epochs of Beta x") # 10
    flags.DEFINE_float("MIN_BETA_X_FAKE", 0.01, "Minimum of beta x") # 0.01
    flags.DEFINE_float("MAX_BETA_X_FAKE", 1, "Maximum of beta x") # 1    
    ## Beta y TRANS
    flags.DEFINE_float("BETA_Y_TRANS", 1, "Coefficient for loss of Y TRANS") # 5
    flags.DEFINE_float("BETA_Y_TRANS_CHANGE_RATE", 1, "Change rate of Beta Y TRANS") # 1
    flags.DEFINE_float("BETA_Y_TRANS_CHANGE_EPOCHS", 100, "Change epochs of Beta Y TRANS") # 100
    flags.DEFINE_float("MIN_BETA_Y_TRANS", 1, "Minimum of beta Y TRANS") # 50
    flags.DEFINE_float("MAX_BETA_Y_TRANS", 1, "Maximum of beta Y TRANS") # 50
    ## Beta y fake
    flags.DEFINE_float("BETA_Y_FAKE", 0, "Coefficient for loss of Y FAKE") # 0
    flags.DEFINE_float("BETA_Y_FAKE_CHANGE_RATE", 1.2, "Change rate of Beta Y FAKE") # 1.2
    flags.DEFINE_float("BETA_Y_FAKE_CHANGE_EPOCHS", 10, "Change epochs of Beta Y FAKE") # 10
    flags.DEFINE_float("MIN_BETA_Y_FAKE", 0, "Minimum of beta Y FAKE") # 50
    flags.DEFINE_float("MAX_BETA_Y_FAKE", 0, "Maximum of beta Y FAKE") # 120
    ## Beta y clean
    flags.DEFINE_float("BETA_Y_CLEAN", 1, "Coefficient for loss of Y CLEAN") # 1
    flags.DEFINE_float("BETA_Y_CLEAN_CHANGE_RATE", 1, "Change rate of Beta Y CLEAN") # 1
    flags.DEFINE_float("BETA_Y_CLEAN_CHANGE_EPOCHS", 100, "Change epochs of Beta Y CLEAN") # 100
    flags.DEFINE_float("MIN_BETA_Y_CLEAN", 1, "Minimum of beta Y CLEAN") # 1
    flags.DEFINE_float("MAX_BETA_Y_CLEAN", 1, "Maximum of beta Y CLEAN") # 1
    # Optimization params
    flags.DEFINE_string("OPT_TYPE", "ADAM", "The type of optimization") # ADAM, MOME, NEST
    flags.DEFINE_float("BATCH_MOME", 0.99, "Momentum for the moving average")
    flags.DEFINE_float("BATCH_EPSILON", 0.001, "Small float added to variance to avoid dividing by zero")
    flags.DEFINE_float("LEARNING_RATE", 1e-4, "Learning rate of optimization")
    flags.DEFINE_float("LEARNING_DECAY_RATE", 0.99, "Decay rate of learning rate")
    flags.DEFINE_integer("LEARNING_DECAY_STEPS", int(2.5*1e3), "Decay steps of learning rate")
    # Non-linear func params
    flags.DEFINE_float("PIXEL_BOUND", 0.01, "Bound for pixel distance") # 0.01
    flags.DEFINE_float("BOUND_CHANGE_RATE", 0.8, "Bound change rate") # 0.8
    flags.DEFINE_float("BOUND_CHANGE_EPOCHS", 2, "Num of epochs per bound change") # 2
    flags.DEFINE_float("MIN_BOUND", 0.001, "Minimum bound for pixel distance") # 4
    flags.DEFINE_float("MAX_BOUND", 1, "Maximum bound for pixel distance") # 128
    # Attack params
    flags.DEFINE_float("EPSILON", 1, "Epsilon for fgm attack") # 128
    flags.DEFINE_float("EPSILON_CHANGE_RATE", 0.8, "Epsilon change rate") # 128
    flags.DEFINE_float("EPSILON_CHANGE_EPOCHS", 100, "Num of epochs for epsilon change") # 100
    flags.DEFINE_float("MIN_EPSILON", 0.1, "Minimum epsilon for fgm attack") # 4
    flags.DEFINE_float("MAX_EPSILON", 128, "Maximum epsilon for fgm attack") # 128
    flags.DEFINE_integer("FGM_ITERS", 1, "Iteration for fgm attack") # 1
    


# Init tensorboard summary writer
def init_writer(LOG_DIR, graph):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    else:
        files = os.listdir(LOG_DIR)
        for log_file in files:
            if log_file.startswith("events"):
                file_path = os.path.join(LOG_DIR, log_file)
                os.remove(file_path)
    writer = tf.summary.FileWriter(LOG_DIR, graph=graph)
    #writer.add_graph(graph)# tensorboard
    return writer