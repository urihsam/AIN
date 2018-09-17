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
    # Loss params
    flags.DEFINE_string("LOSS_MODE", "ENTRO", "How to calculate loss") # LOGITS, PREDS, ENTRO
    flags.DEFINE_float("BETA_X", 0.1, "Coefficient for loss of X") # 0.1
    flags.DEFINE_float("BETA_Y", 50, "Coefficient for loss of Y") # 50
    flags.DEFINE_string("NORM_TYPE", "L2", "The norm type") # INF, L2, L1
    flags.DEFINE_bool("PARTIAL_LOSS", False, "Use partial loss or not")
    flags.DEFINE_float("PARTIAL_THRESHOLD", 0.2, "The threshold for partial loss switch")
    flags.DEFINE_float('REG_SCALE', 0.01, 'The scale of regularization')
    # Optimization params
    flags.DEFINE_string("OPT_TYPE", "ADAM", "The type of optimization") # ADAM, MOME, NEST
    flags.DEFINE_float("BATCH_MOME", 0.99, "Momentum for the moving average")
    flags.DEFINE_float("BATCH_EPSILON", 0.001, "Small float added to variance to avoid dividing by zero")
    flags.DEFINE_float("LEARNING_RATE", 1e-4, "Learning rate of optimization")
    flags.DEFINE_float("LEARNING_DECAY_RATE", 0.99, "Decay rate of learning rate")
    flags.DEFINE_integer("LEARNING_DECAY_STEPS", int(2.5*1e3), "Decay steps of learning rate")
    # Non-linear func params
    flags.DEFINE_float("PIXEL_BOUND", 16, "Bound for pixel distance") # 16
    flags.DEFINE_float("BOUND_DECAY_RATE", 0.8, "Bound decay rate") # 0.8
    flags.DEFINE_float("BOUND_DECAY_EPOCHS", 2, "Num of epochs per bound decay") # 2
    flags.DEFINE_float("MIN_BOUND", 4, "Minimum bound for pixel distance") # 4
    # Attack params
    flags.DEFINE_float("EPSILON", 128, "Epsilon for fgm attack") # 128
    flags.DEFINE_float("EPSILON_DECAY_RATE", 0.8, "Epsilon decay rate") # 128
    flags.DEFINE_float("EPSILON_DECAY_EPOCHS", 2, "Num of epochs for epsilon decay") # 2
    flags.DEFINE_float("MIN_EPSILON", 4, "Minimum epsilon for fgm attack") # 4
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