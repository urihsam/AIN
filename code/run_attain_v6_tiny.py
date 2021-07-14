import os, math, time
import nn.attain_v6_tiny as attain
from PIL import Image
from dependency import *
import utils.model_utils as  model_utils
#from utils.data_utils_mnist_raw import dataset
from utils.data_utils import dataset

model_utils.set_flags()

data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED, 
    adv_path_prefix=FLAGS.ADV_PATH_PREFIX)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_INDEX


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test_info(sess, model, test_writer, graph_dict, log_file, total_batch=None, valid=False):
    model_loss_x, (model_lx_true, model_lx_fake), (model_lx_dist_true, model_lx_dist_fake, model_lx_dist_trans), \
            (model_max_dist_true, model_max_dist_fake, model_max_dist_trans) = model.loss_x(graph_dict["beta_x_t_holder"], graph_dict["beta_x_f_holder"])
    model_loss_y, (model_ly_trans, model_ly_fake, model_ly_clean), (model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean) =\
        model.loss_y(graph_dict["beta_y_t_holder"], graph_dict["beta_y_f_holder"],  graph_dict["beta_y_f2_holder"], graph_dict["beta_y_c_holder"]
                    )
    model_loss, model_recon_loss, model_label_loss, model_sparse_loss, model_var_loss, model_reg = \
        model.loss(graph_dict["partial_loss_holder"], model_loss_x, model_loss_y)
    fetches = [model._target_accuracy, model._target_adv_accuracy, model._target_fake_accuracy, 
            model_loss, model_recon_loss, model_label_loss, model_sparse_loss, model_var_loss, model_reg, 
            model_loss_x, model_lx_true, model_lx_fake, 
            model_lx_dist_true, model_lx_dist_fake, model_lx_dist_trans,
            model_max_dist_true, model_max_dist_fake, model_max_dist_trans,
            model_loss_y, model_ly_trans, model_ly_fake, model_ly_clean, 
            model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean,
            graph_dict["merged_summary"]]
    if total_batch is None:
        if valid:
            total_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
        else:
            total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    else: total_batch = total_batch

    acc = 0; adv_acc = 0; fake_acc = 0; 
    loss = 0; recon_loss = 0; label_loss = 0; sparse_loss = 0; var_loss = 0; reg = 0;
    l_x = 0; Lx_true = 0; Lx_fake = 0; Lx_dist_true = 0; Lx_dist_fake = 0; Lx_dist_trans = 0;
    max_dist_true = 0; max_dist_fake = 0; max_dist_trans = 0;
    l_y = 0; Ly_trans = 0; Ly_fake = 0; Ly_clean = 0; 
    Ly_dist_trans = 0; Ly_dist_fake = 0; Ly_dist_clean = 0
    
    diff = []
    ys = []
    for idx in range(total_batch):
        if valid:
            batch_xs, batch_ys, batch_atks = data.next_valid_batch(FLAGS.BATCH_SIZE, False)
        else:
            batch_xs, batch_ys, batch_atks = data.next_test_batch(FLAGS.BATCH_SIZE, False)
        feed_dict = {
            graph_dict["images_holder"]: batch_xs,
            graph_dict["label_holder"]: batch_ys,
            graph_dict["atk_holder"]: batch_atks,
            graph_dict["low_bound_holder"]: -1.0*FLAGS.PIXEL_BOUND,
            graph_dict["up_bound_holder"]: 1.0*FLAGS.PIXEL_BOUND,
            graph_dict["epsilon_holder"]: FLAGS.EPSILON,
            graph_dict["beta_x_t_holder"]: FLAGS.BETA_X_TRUE,
            graph_dict["beta_x_f_holder"]: FLAGS.BETA_X_FAKE,
            graph_dict["beta_y_t_holder"]: FLAGS.BETA_Y_TRANS,
            graph_dict["beta_y_f_holder"]: FLAGS.BETA_Y_FAKE,
            graph_dict["beta_y_f2_holder"]: FLAGS.BETA_Y_FAKE2,
            graph_dict["beta_y_c_holder"]: FLAGS.BETA_Y_CLEAN,
            graph_dict["partial_loss_holder"]: FLAGS.PARTIAL_LOSS,
            graph_dict["is_training"]: False
        }
        if FLAGS.IS_TARGETED_ATTACK:
            batch_tgt_label = np.asarray(model_utils._one_hot_encode(
                [int(FLAGS.TARGETED_LABEL)]*FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES))
            feed_dict[graph_dict["tgt_label_holder"]] = batch_tgt_label

            
        
        
        batch_acc, batch_adv_acc, batch_fake_acc, batch_loss, batch_recon_loss, batch_label_loss, batch_sparse_loss, batch_var_loss, batch_reg, \
            batch_l_x, batch_lx_true, batch_lx_fake, batch_Lx_dist_true, batch_Lx_dist_fake, batch_Lx_dist_trans, \
            batch_max_dist_true, batch_max_dist_fake, batch_max_dist_trans, \
            batch_l_y, batch_Ly_trans, batch_Ly_fake, batch_Ly_clean,\
            batch_Ly_dist_trans, batch_Ly_dist_fake, batch_Ly_dist_clean, \
            summary = sess.run(fetches=fetches, feed_dict=feed_dict)
        if FLAGS.SAVE_DIFF:
            batch_adv = sess.run(fetches=model.prediction, feed_dict=feed_dict)
            diff.append(batch_adv-batch_xs)
            ys.append(batch_ys)
        #test_writer.add_summary(summary, idx)
        acc += batch_acc
        adv_acc += batch_adv_acc
        fake_acc += batch_fake_acc
        loss += batch_loss
        recon_loss += batch_recon_loss
        label_loss += batch_label_loss
        sparse_loss += batch_sparse_loss
        var_loss += batch_var_loss
        reg += batch_reg
        l_x += batch_l_x
        Lx_true += batch_lx_true
        Lx_fake += batch_lx_fake
        Lx_dist_true += batch_Lx_dist_true
        Lx_dist_fake += batch_Lx_dist_fake
        Lx_dist_trans += batch_Lx_dist_trans
        max_dist_true += batch_max_dist_true
        max_dist_fake += batch_max_dist_fake
        max_dist_trans += batch_max_dist_trans
        l_y += batch_l_y
        Ly_trans += batch_Ly_trans
        Ly_fake += batch_Ly_fake
        Ly_clean += batch_Ly_clean
        Ly_dist_trans += batch_Ly_dist_trans
        Ly_dist_fake += batch_Ly_dist_fake
        Ly_dist_clean += batch_Ly_dist_clean
        
        
    acc /= total_batch
    adv_acc /= total_batch
    fake_acc /= total_batch
    loss /= total_batch
    recon_loss /= total_batch
    label_loss /= total_batch
    sparse_loss /= total_batch
    var_loss /= total_batch
    reg /= total_batch
    l_x /= total_batch
    Lx_true /= total_batch
    Lx_fake /= total_batch
    Lx_dist_true /= total_batch
    Lx_dist_fake /= total_batch
    Lx_dist_trans /= total_batch
    max_dist_true /= total_batch
    max_dist_fake /= total_batch
    max_dist_trans /= total_batch
    l_y /= total_batch
    Ly_trans /= total_batch
    Ly_fake /= total_batch
    Ly_clean /= total_batch
    Ly_dist_trans /= total_batch
    Ly_dist_fake /= total_batch
    Ly_dist_clean /= total_batch
    if FLAGS.SAVE_DIFF:
        diff = np.concatenate(diff, 0)
        ys = np.concatenate(ys, 0)
        if FLAGS.IS_TARGETED_ATTACK:
            np.save("AIN_tiny_tgt_ys_diversity.npy", ys)
            np.save("AIN_tiny_tgt_diversity.npy", diff)
        else:
            np.save("AIN_tiny_untgt_ys_diversity.npy", ys)
            np.save("AIN_tiny_untgt_diversity.npy", diff)

    print('Original accuracy: {0:0.5f}'.format(acc))
    print('Faked accuracy: {0:0.5f}'.format(fake_acc))
    print('Attacked accuracy: {0:0.5f}'.format(adv_acc))
    print("Loss = {:.4f}  Loss recon = {:.4f}  Loss label = {:.4f} Loss sparse = {:.4f}  Loss var = {:.4f}  Loss reg = {:.4f}".format(
        loss, recon_loss, label_loss, sparse_loss, var_loss, reg))
    print("Loss x = {:.4f}  Loss y = {:.4f}".format(l_x, l_y))
    print("Loss x for true = {:.4f} Loss x for fake = {:.4}".format(Lx_true, Lx_fake))
    print("Loss y for trans = {:.4f} Loss y for fake = {:.4f} Loss y for clean = {:.4f}".format(Ly_trans, Ly_fake, Ly_clean))
    print("Lx distance for true = {:.4f} Lx distance for fake = {:.4f} Lx distance for trans = {:.4f}".format(Lx_dist_true, Lx_dist_fake, Lx_dist_trans))
    print("Max pixel distance for true = {:.4f} Max pixel distance for fake = {:.4f} Max pixel distance for trans = {:.4f}".format(
        max_dist_true, max_dist_fake, max_dist_trans))
    print("Ly distance for trans = {:.4f} Ly distance for fake = {:.4f} Ly distance for clean = {:.4f}".format(
        Ly_dist_trans, Ly_dist_fake, Ly_dist_clean))
    
    with open(log_file, "a+") as file: 
        file.write('Original accuracy: {0:0.5f} \n'.format(acc))
        file.write('Faked accuracy: {0:0.5f} \n'.format(fake_acc))
        file.write('Attacked accuracy: {0:0.5f} \n'.format(adv_acc))
        file.write("Loss = {:.4f}  Loss recon = {:.4f}  Loss label = {:.4f} Loss sparse = {:.4f}  Loss var = {:.4f}  Loss reg = {:.4f}\n".format(
            loss, recon_loss, label_loss, sparse_loss, var_loss, reg))
        file.write("Loss x = {:.4f}  Loss y = {:.4f}\n".format(l_x, l_y))
        file.write("Loss x for true = {:.4f} Loss x for fake = {:.4}\n".format(Lx_true, Lx_fake))
        file.write("Loss y for trans = {:.4f} Loss y for fake = {:.4f} Loss y for clean = {:.4f}\n".format(Ly_trans, Ly_fake, Ly_clean))
        file.write("Lx distance for true = {:.4f} Lx distance for fake = {:.4f} Lx distance for trans = {:.4f}\n".format(Lx_dist_true, Lx_dist_fake, Lx_dist_trans))
        file.write("Max pixel distance for true = {:.4f} Max pixel distance for fake = {:.4f} Max pixel distance for trans = {:.4f}\n".format(
            max_dist_true, max_dist_fake, max_dist_trans))
        file.write("Ly distance for trans = {:.4f} Ly distance for fake = {:.4f} Ly distance for clean = {:.4f} \n".format(
            Ly_dist_trans, Ly_dist_fake, Ly_dist_clean))
        file.write("############################################\n")
        file.flush()
    
    res_dict = {"acc": acc, 
                "fake_acc": fake_acc,
                "adv_acc": adv_acc, 
                "loss": loss, 
                "recon_loss": recon_loss,
                "label_loss": label_loss,
                "sparse_loss": sparse_loss,
                "var_loss": var_loss,
                "l_x": l_x,
                "Lx_true": Lx_true,
                "Lx_fake": Lx_fake,
                "Lx_dist_true": Lx_dist_true,
                "Lx_dist_fake": Lx_dist_fake,
                "Lx_dist_trans": Lx_dist_trans,
                "max_dist_true": max_dist_true,
                "max_dist_fake": max_dist_fake,
                "max_dist_trans": max_dist_trans,
                "l_y": l_y, 
                "Ly_trans": Ly_trans,
                "Ly_fake": Ly_fake,
                "Ly_clean": Ly_clean,
                "Ly_dist_trans": Ly_dist_trans,
                "Ly_dist_fake": Ly_dist_fake,
                "Ly_dist_clean": Ly_dist_clean
                }
    return res_dict



def test():
    """
    """
    tf.reset_default_graph()
    g = tf.get_default_graph()

    with g.as_default():
        # Placeholder nodes.
        images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        atk_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        tgt_label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        low_bound_holder = tf.placeholder(tf.float32, ())
        up_bound_holder = tf.placeholder(tf.float32, ())
        epsilon_holder = tf.placeholder(tf.float32, ())
        beta_x_t_holder = tf.placeholder(tf.float32, ())
        beta_x_f_holder = tf.placeholder(tf.float32, ())
        beta_y_t_holder = tf.placeholder(tf.float32, ())
        beta_y_f_holder = tf.placeholder(tf.float32, ())
        beta_y_f2_holder = tf.placeholder(tf.float32, ())
        beta_y_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.string, ())
        is_training = tf.placeholder(tf.bool, ())

        if FLAGS.IS_TARGETED_ATTACK:
            model = attain.ATTAIN(images_holder, label_holder, low_bound_holder, up_bound_holder, 
                              epsilon_holder, is_training, 
                              targeted_label=tgt_label_holder)
        else:
            model = attain.ATTAIN(images_holder, label_holder, low_bound_holder, up_bound_holder, 
                              epsilon_holder, is_training)

        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["atk_holder"] = atk_holder
        graph_dict["tgt_label_holder"] = tgt_label_holder
        graph_dict["low_bound_holder"] = low_bound_holder
        graph_dict["up_bound_holder"] = up_bound_holder
        graph_dict["epsilon_holder"] = epsilon_holder
        graph_dict["beta_x_t_holder"] = beta_x_t_holder
        graph_dict["beta_x_f_holder"] = beta_x_f_holder
        graph_dict["beta_y_t_holder"] = beta_y_t_holder
        graph_dict["beta_y_f_holder"] = beta_y_f_holder
        graph_dict["beta_y_f2_holder"] = beta_y_f2_holder
        graph_dict["beta_y_c_holder"] = beta_y_c_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary
        

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # Load target classifier
        model._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
        model.tf_load(sess, name=FLAGS.AE_CKPT_RESTORE_NAME)
        model.tf_load(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")
        # tensorboard writer
        #test_writer = model_utils.init_writer(FLAGS.TEST_LOG_PATH, g)
        print("\nTest")
        if FLAGS.local:
            total_test_batch = 2
        else:
            total_test_batch = None
        if FLAGS.USE_IMITATION == False:
            postfix = "_no_imit"
        elif FLAGS.ONLY_IMITATION:
            postfix = "_only_imit"
        elif FLAGS.USE_DISTANCE == False:
            postfix = "_no_dist"
        elif FLAGS.ONLY_DISTANCE:
            postfix = "_only_dist"
        elif FLAGS.USE_MISCLASSIFY == False:
            postfix = "_no_misc"
        elif FLAGS.ONLY_MISCLASSIFY:
            postfix = "_only_misc"
        elif FLAGS.USE_ATT == False:
            postfix = "_no_att"
        elif FLAGS.NUM_ENC_RES_BLOCK == 0 and FLAGS.ENC_RES_BLOCK_SIZE == 0:
            postfix = "_no_res"
        elif FLAGS.LABEL_CONDITIONING == False:
            postfix = "_no_lc"
        else: 
            postfix = ""
        if FLAGS.IS_TARGETED_ATTACK:
            log_file_name = "tgt{}_tiny_test_log{}.txt".format(FLAGS.TARGETED_LABEL, postfix)
        else:
            log_file_name = "untgt_tiny_test_log{}.txt".format(postfix)
        test_info(sess, model, None, graph_dict, log_file_name, total_batch=total_test_batch)
        #test_writer.close() 
        
        
        size = 50
        batch_xs, batch_ys, _ = data.next_test_batch(size, True)
        targeted_label = np.asarray(model_utils._one_hot_encode(
                            [int(FLAGS.TARGETED_LABEL)]*size, FLAGS.NUM_CLASSES))
        feed_dict = {
            images_holder: batch_xs,
            label_holder: batch_ys,
            low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
            up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
            is_training: False
        }
        if FLAGS.IS_TARGETED_ATTACK:
            feed_dict[tgt_label_holder]= np.asarray(model_utils._one_hot_encode(
                                [int(FLAGS.TARGETED_LABEL)]*10, FLAGS.NUM_CLASSES))
        # attack
        start = time.time()
        adv_images = sess.run(fetches=model.prediction, feed_dict=feed_dict)
        time_cost = (time.time() - start)
        l_inf = np.mean(
            np.amax(
                np.absolute(np.reshape(adv_images, (size, 64*64*3))-np.reshape(batch_xs, (size, 64*64*3))), 
                axis=-1)
            )
        
        l_2 = np.mean(
            np.sqrt(np.sum(
                np.square(np.reshape(adv_images, (size, 64*64*3))-np.reshape(batch_xs, (size, 64*64*3))), 
                axis=-1)
            ))
    
        print("L inf: {}".format(l_inf))
        print("L 2: {}".format(l_2))
        print("Time cost:", time_cost/size)

        batch_xs = np.load("tiny_plot_examples.npy")/255.0
        batch_ys = np.load("tiny_plot_example_labels.npy")
        feed_dict = {
            images_holder: batch_xs,
            label_holder: batch_ys,
            low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
            up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
            is_training: False
        }
        if FLAGS.IS_TARGETED_ATTACK:
            feed_dict[tgt_label_holder] = np.asarray(model_utils._one_hot_encode(
                                [int(FLAGS.TARGETED_LABEL)]*10, FLAGS.NUM_CLASSES))
        adv_images = sess.run(model.prediction, feed_dict=feed_dict)
        
        width = 10*64
        height = 2*64
        new_im = Image.new('RGB', (width, height))
        x_offset = 0
        y_offset = 64
        for i in range(10):
            im1 = Image.fromarray(np.uint8(batch_xs[i]*255.0))
            im2 = Image.fromarray(np.uint8(adv_images[i]*255.0))
            new_im.paste(im1, (x_offset, 0))
            new_im.paste(im2, (x_offset, y_offset))
            x_offset += im1.size[0]

        new_im.show()
        if FLAGS.USE_IMITATION == False:
            postfix = "_no_imit"
        elif FLAGS.ONLY_IMITATION:
            postfix = "_only_imit"
        elif FLAGS.USE_DISTANCE == False:
            postfix = "_no_dist"
        elif FLAGS.ONLY_DISTANCE:
            postfix = "_only_dist"
        elif FLAGS.USE_MISCLASSIFY == False:
            postfix = "_no_misc"
        elif FLAGS.ONLY_MISCLASSIFY:
            postfix = "_only_misc"
        elif FLAGS.USE_ATT == False:
            postfix = "_no_att"
        elif FLAGS.NUM_ENC_RES_BLOCK == 0 and FLAGS.ENC_RES_BLOCK_SIZE == 0:
            postfix = "_no_res"
        elif FLAGS.LABEL_CONDITIONING == False:
            postfix = "_no_lc"
        else: 
            postfix = ""
        if FLAGS.IS_TARGETED_ATTACK:
            img_name = "AIN_TINY_TGT{}{}.jpg".format(FLAGS.TARGETED_LABEL, postfix)
        else:
            img_name = "AIN_TINY_UNTGT{}.jpg".format(postfix)
        new_im.save(img_name)
        
        


def train():
    """
    """
    INIT_PIXEL_BOUND = FLAGS.PIXEL_BOUND
    INIT_BOUND_CHANGE_RATE = FLAGS.BOUND_CHANGE_RATE
    INIT_BETA_X_TRUE = FLAGS.BETA_X_TRUE
    INIT_BETA_X_FAKE = FLAGS.BETA_X_FAKE
    INIT_BETA_Y_TRANS = FLAGS.BETA_Y_TRANS
    INIT_BETA_Y_FAKE = FLAGS.BETA_Y_FAKE
    INIT_BETA_Y_FAKE2 = FLAGS.BETA_Y_FAKE2
    INIT_BETA_Y_CLEAN = FLAGS.BETA_Y_CLEAN
    import time
    tf.reset_default_graph()
    g = tf.get_default_graph()
    # attack_target = 8
    with g.as_default():
        # Placeholder nodes.
        images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        atk_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        tgt_label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        low_bound_holder = tf.placeholder(tf.float32, ())
        up_bound_holder = tf.placeholder(tf.float32, ())
        epsilon_holder = tf.placeholder(tf.float32, ())
        beta_x_t_holder = tf.placeholder(tf.float32, ())
        beta_x_f_holder = tf.placeholder(tf.float32, ())
        beta_y_t_holder = tf.placeholder(tf.float32, ())
        beta_y_f_holder = tf.placeholder(tf.float32, ())
        beta_y_f2_holder = tf.placeholder(tf.float32, ())
        beta_y_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.string, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        if FLAGS.IS_TARGETED_ATTACK:
            model = attain.ATTAIN(images_holder, label_holder, low_bound_holder, up_bound_holder, 
                              epsilon_holder, is_training, atk_holder, tgt_label_holder)
        else:
            model = attain.ATTAIN(images_holder, label_holder, low_bound_holder, up_bound_holder, 
                              epsilon_holder, is_training, atk_holder)

        # pre-training
        pre_label_loss = FLAGS.GAMMA_PRE_L * model.pre_loss_label
        pre_op, _, _, _ = model.optimization(pre_label_loss, scope="PRE_OPT")
        # model training
        
        model_loss_x, (model_lx_true, model_lx_fake), (model_lx_dist_true, model_lx_dist_fake, model_lx_dist_trans), \
            (model_max_dist_true, model_max_dist_fake, model_max_dist_trans) = model.loss_x(beta_x_t_holder, beta_x_f_holder)
        model_loss_y, (model_ly_trans, model_ly_fake, model_ly_clean), (model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean) =\
            model.loss_y(beta_y_t_holder, beta_y_f_holder, beta_y_f2_holder, beta_y_c_holder
                        )
        model_loss, model_recon_loss, model_label_loss, model_sparse_loss, model_var_loss, model_reg = \
            model.loss(partial_loss_holder, model_loss_x, model_loss_y)
        model_op, (model_zero_op,  model_accum_op, model_avg_op), model_lr_reset_op, model_lr = model.optimization(model_loss, accum_iters=FLAGS.NUM_ACCUM_ITERS)
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["atk_holder"] = atk_holder
        graph_dict["tgt_label_holder"] = tgt_label_holder
        graph_dict["low_bound_holder"] = low_bound_holder
        graph_dict["up_bound_holder"] = up_bound_holder
        graph_dict["epsilon_holder"] = epsilon_holder
        graph_dict["beta_x_t_holder"] = beta_x_t_holder
        graph_dict["beta_x_f_holder"] = beta_x_f_holder
        graph_dict["beta_y_t_holder"] = beta_y_t_holder
        graph_dict["beta_y_f_holder"] = beta_y_f_holder
        graph_dict["beta_y_f2_holder"] = beta_y_f2_holder
        graph_dict["beta_y_c_holder"] = beta_y_c_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # Load target classifier
        model._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
        if FLAGS.load_AE:
            print("Autoencoder loaded.")
            model.tf_load(sess, name=FLAGS.AE_CKPT_RESTORE_NAME)
            #model.tf_load(sess, name='deep_cae_last.ckpt')
            model.tf_load(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")
        # For tensorboard
        train_writer = model_utils.init_writer(FLAGS.TRAIN_LOG_PATH, g)
        valid_writer = model_utils.init_writer(FLAGS.VALID_LOG_PATH, g)
        
        if FLAGS.local:
            total_train_batch = 2
            total_pre_train_batch = 2
            total_valid_batch = 2
            total_pre_valid_batch = 2
        else:
            total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE/FLAGS.NUM_ACCUM_ITERS)
            total_pre_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
            total_valid_batch = None
            total_pre_valid_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
        

        if FLAGS.train_label:
            print("Pre-training...")
            for epoch in range(FLAGS.NUM_PRE_EPOCHS):
                start_time = time.time()
                for train_idx in range(total_pre_train_batch):
                    batch_xs, batch_ys, _ = data.next_train_batch(FLAGS.BATCH_SIZE, True)
                    feed_dict = {
                        label_holder: batch_ys,
                        is_training: True
                    }
                    fetches = [pre_op, pre_label_loss]
                    _, pre_loss = sess.run(fetches=fetches, feed_dict=feed_dict)

                if epoch % FLAGS.PRE_EVAL_FREQUENCY == (FLAGS.PRE_EVAL_FREQUENCY - 1):
                    print("Training Result:")
                    print("Pre Loss label = {:.4f}".format(pre_loss))
                    pre_valid_loss = 0
                    for valid_idx in range(total_pre_valid_batch):
                        batch_xs, batch_ys, _ = data.next_valid_batch(FLAGS.BATCH_SIZE, True)
                        feed_dict = {
                            label_holder: batch_ys,
                            is_training: True
                        }
                        fetches = pre_label_loss
                        pre_valid_loss += sess.run(fetches=fetches, feed_dict=feed_dict)
                    pre_valid_loss /= total_pre_valid_batch
                    print("============================")
                    print("Validation Result:")
                    print("Pre Loss label = {:.4f}".format(pre_valid_loss))
                    print("============================")
            print("Save label states into ckpt...")
            model.tf_save(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")
        else:
            print("Loading label states from ckpt...")
            model.tf_load(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")
        
        # reset learning rate
        sess.run(fetches=[model_lr_reset_op])
        print("Training...")
        prev_valid_acc = 0.0
        last_prev_valid_acc = 0.0
        change_itr = 0
        last_change_itr = 0
        last_pixel_bound = INIT_PIXEL_BOUND
        last_beta_x_true = INIT_BETA_X_TRUE
        last_beta_x_fake = INIT_BETA_X_FAKE
        last_beta_y_trans = INIT_BETA_Y_TRANS 
        last_beta_y_fake = INIT_BETA_Y_FAKE
        last_beta_y_fake2 = INIT_BETA_Y_FAKE2
        last_beta_y_clean = INIT_BETA_Y_CLEAN
        ###
        min_lx_dist = np.inf
        last_min_lx_dist = np.inf
        ###
        roll_back_pixel = INIT_PIXEL_BOUND
        roll_back_hit = 0
        if FLAGS.IS_TARGETED_ATTACK:
            max_valid_acc = FLAGS.INIT_MAX_VALID_ACC
            last_max_valid_acc = 0.0
        else:
            min_valid_acc = FLAGS.INIT_MIN_VALID_ACC
            last_min_valid_acc = 1.0
        for epoch in range(FLAGS.NUM_EPOCHS):
            start_time = time.time()
            for train_idx in range(total_train_batch):
                if FLAGS.NUM_ACCUM_ITERS != 1:
                    for accum_idx in range(FLAGS.NUM_ACCUM_ITERS):
                        batch_xs, batch_ys, batch_atks = data.next_train_batch(FLAGS.BATCH_SIZE, False)
                        feed_dict = {
                            images_holder: batch_xs,
                            label_holder: batch_ys,
                            atk_holder: batch_atks,
                            low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
                            up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
                            epsilon_holder: FLAGS.EPSILON,
                            beta_x_t_holder: FLAGS.BETA_X_TRUE,
                            beta_x_f_holder: FLAGS.BETA_X_FAKE,
                            beta_y_t_holder: FLAGS.BETA_Y_TRANS,
                            beta_y_f_holder: FLAGS.BETA_Y_FAKE,
                            beta_y_f2_holder: FLAGS.BETA_Y_FAKE2,
                            beta_y_c_holder: FLAGS.BETA_Y_CLEAN,
                            is_training: True,
                            partial_loss_holder: FLAGS.PARTIAL_LOSS
                        }
                        if FLAGS.IS_TARGETED_ATTACK:
                            batch_tgt_label = np.asarray(model_utils._one_hot_encode(
                                [int(FLAGS.TARGETED_LABEL)]*FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES))
                            feed_dict[tgt_label_holder] = batch_tgt_label
                        
                        sess.run(fetches=[model_accum_op], feed_dict=feed_dict)
                    sess.run(fetches=[model_avg_op])
                
                else:
                    batch_xs, batch_ys, batch_atks = data.next_train_batch(FLAGS.BATCH_SIZE, False)
                    feed_dict = {
                        images_holder: batch_xs,
                        label_holder: batch_ys,
                        atk_holder: batch_atks,
                        low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
                        up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
                        epsilon_holder: FLAGS.EPSILON,
                        beta_x_t_holder: FLAGS.BETA_X_TRUE,
                        beta_x_f_holder: FLAGS.BETA_X_FAKE,
                        beta_y_t_holder: FLAGS.BETA_Y_TRANS,
                        beta_y_f_holder: FLAGS.BETA_Y_FAKE,
                        beta_y_f2_holder: FLAGS.BETA_Y_FAKE2,
                        beta_y_c_holder: FLAGS.BETA_Y_CLEAN,
                        #kappa_t_holder: FLAGS.KAPPA_FOR_TRANS,
                        #kappa_f_holder: FLAGS.KAPPA_FOR_FAKE,
                        #kappa_c_holder: FLAGS.KAPPA_FOR_CLEAN,
                        is_training: True,
                        partial_loss_holder: FLAGS.PARTIAL_LOSS
                    }
                    if FLAGS.IS_TARGETED_ATTACK:
                        batch_tgt_label = np.asarray(model_utils._one_hot_encode(
                            [int(FLAGS.TARGETED_LABEL)]*FLAGS.BATCH_SIZE, FLAGS.NUM_CLASSES))
                        feed_dict[tgt_label_holder] = batch_tgt_label
                    
                """[res0, res1, res2] = sess.run([model.adv, model.fake, model.cross_entropy],
                                        feed_dict=feed_dict)
                import pdb; pdb.set_trace()"""
                # optimization
                fetches = [model_op, model_loss, 
                        model_recon_loss, model_label_loss, model_sparse_loss, model_var_loss, model_reg, 
                        model_loss_x, model_lx_true, model_lx_fake, 
                        model_lx_dist_true, model_lx_dist_fake, model_lx_dist_trans,
                        model_max_dist_true, model_max_dist_fake, model_max_dist_trans,
                        model_loss_y, model_ly_trans, model_ly_fake, model_ly_clean, 
                        model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean,
                        merged_summary, model._target_fake_prediction,
                        model_lr, model._encoder.att_gamma, model._decoder_t.att_gamma,
                        model._target_accuracy, model._target_adv_accuracy, model._target_fake_accuracy]
                _, loss, recon_loss, label_loss, sparse_loss, var_loss, reg, l_x, Lx_true, Lx_fake, Lx_dist_true, Lx_dist_fake, Lx_dist_trans, \
                    max_dist_true, max_dist_fake, max_dist_trans, l_y, Ly_trans, Ly_fake, Ly_clean, \
                    Ly_dist_trans, Ly_dist_fake, Ly_dist_clean, \
                    summary, fake_prediction,\
                    lr, enc_att_gamma, dec_att_gamma,\
                    clean_acc, adv_acc, fake_acc = sess.run(fetches=fetches, feed_dict=feed_dict)
                
                #import pdb; pdb.set_trace()
                #train_writer.add_summary(summary, train_idx)
                # Print info
                if train_idx % FLAGS.EVAL_FREQUENCY == (FLAGS.EVAL_FREQUENCY - 1):
                    print("Epoch: {}".format(epoch+1))
                    print("Learning rate: {}".format(lr))
                    print("Clean Accuracy: {:.4f}, Adv Accuracy: {:.4f}, Fake Accuracy: {:.4f}".format(clean_acc, adv_acc, fake_acc))
                    print("Enc Att Gamma: {} ; Dec Att Gamma: {}".format(enc_att_gamma, dec_att_gamma))
                    print("Hyper-params info:")
                    print("Using Partial Loss:", FLAGS.PARTIAL_LOSS)
                    print("Pixel bound: [{:.4f}, {:.4f}]  Epsilon: {:.4f}".format(
                        -1.0*FLAGS.PIXEL_BOUND, FLAGS.PIXEL_BOUND, FLAGS.EPSILON))
                    print("BETA_X_TRUE: {:.4f}  BETA_X_FAKE: {:.4f}   Beta_Y_TRANS: {:.4f}  Beta_Y_FAKE: {:.4f}  Beta_Y_CLEAN: {:.4f}".format(
                        FLAGS.BETA_X_TRUE, FLAGS.BETA_X_FAKE, FLAGS.BETA_Y_TRANS, FLAGS.BETA_Y_FAKE, FLAGS.BETA_Y_CLEAN
                    ))
                    print("Result:")
                    print("Loss = {:.4f}  Loss recon = {:.4f}  Loss label = {:.4f}  Loss sparse = {:.4f}  Loss var = {:.4f}  Loss reg = {:.4f}".format(
                        loss, recon_loss, label_loss, sparse_loss, var_loss, reg))
                    print("Loss x = {:.4f}  Loss y = {:.4f}".format(l_x, l_y))
                    print("Loss x for true = {:.4f} Loss x for fake = {:.4}".format(Lx_true, Lx_fake))
                    print("Loss y for trans = {:.4f} Loss y for fake = {:.4f} Loss y for clean = {:.4f}".format(Ly_trans, Ly_fake, Ly_clean))
                    print("Lx distance for true = {:.4f} Lx distance for fake = {:.4f} Lx distance for trans = {:.4f}".format(Lx_dist_true, Lx_dist_fake, Lx_dist_trans))
                    print("Max pixel distance for true = {:.4f} Max pixel distance for fake = {:.4f} Max pixel distance for trans = {:.4f}".format(
                        max_dist_true, max_dist_fake, max_dist_trans))
                    print("Ly distance for trans = {:.4f} Ly distance for fake = {:.4f} Ly distance for clean = {:.4f}".format(
                        Ly_dist_trans, Ly_dist_fake, Ly_dist_clean))
                    print()
                    #model.tf_save(sess) # save checkpoint
                    
                if FLAGS.PARTIAL_LOSS != "FULL_LOSS":
                    if train_idx % FLAGS.LOSS_CHANGE_FREQUENCY == (FLAGS.LOSS_CHANGE_FREQUENCY - 1):
                        if Lx_dist <= FLAGS.LOSS_X_THRESHOLD:
                            FLAGS.PARTIAL_LOSS = "LOSS_Y"
                        else:
                            FLAGS.PARTIAL_LOSS = "LOSS_X"
                      
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
            # validation
            if epoch % FLAGS.VALID_FREQUENCY == FLAGS.VALID_FREQUENCY-1:
                print("\n******************************************************************")
                print("Validation")
                if FLAGS.USE_IMITATION == False:
                    postfix = "_no_imit"
                elif FLAGS.ONLY_IMITATION:
                    postfix = "_only_imit"
                elif FLAGS.USE_DISTANCE == False:
                    postfix = "_no_dist"
                elif FLAGS.ONLY_DISTANCE:
                    postfix = "_only_dist"
                elif FLAGS.USE_MISCLASSIFY == False:
                    postfix = "_no_misc"
                elif FLAGS.ONLY_MISCLASSIFY:
                    postfix = "_only_misc"
                elif FLAGS.USE_ATT == False:
                    postfix = "_no_att"
                elif FLAGS.NUM_ENC_RES_BLOCK == 0 and FLAGS.ENC_RES_BLOCK_SIZE == 0:
                    postfix = "_no_res"
                elif FLAGS.LABEL_CONDITIONING == False:
                    postfix = "_no_lc"
                else: 
                    postfix = ""
                if FLAGS.IS_TARGETED_ATTACK:
                    valid_log_name = "tgt{}_tiny_valid_log{}.txt".format(FLAGS.TARGETED_LABEL, postfix)
                else:
                    valid_log_name = "untgt_tiny_valid_log{}.txt".format(postfix)
                valid_dict = test_info(sess, model, None, graph_dict, valid_log_name, total_batch=total_valid_batch, valid=True)
                

                #if valid_dict["adv_acc"] > valid_dict["fake_acc"] and valid_dict["adv_acc"] > 0.1: # stop
                #if valid_dict["max_dist_true"] <= valid_dict["max_dist_trans"]: # stop
                #break
                #else:
                ckpt_name='deep_cae.Linf{:.6f}.Lx{:.6f}.acc{:.6f}.ckpt'.format(
                    valid_dict["max_dist_true"],
                    valid_dict["Lx_dist_true"],
                    valid_dict["adv_acc"]
                    )
                if FLAGS.IS_TARGETED_ATTACK:
                    if valid_dict["adv_acc"] >= max_valid_acc:
                        if valid_dict["Lx_dist_true"] <= min_lx_dist:
                            print("Find model at bound step {} has larger valid acc: {}".format(FLAGS.PIXEL_BOUND, valid_dict["adv_acc"]))
                            model.tf_save(sess, name=ckpt_name) # extra store
                            model.tf_save(sess)
                            max_valid_acc = valid_dict["adv_acc"]
                            min_lx_dist = valid_dict["Lx_dist_true"]
                            print("Trained params have been saved to '%s'" % FLAGS.AE_PATH)
                else:
                    if valid_dict["adv_acc"] <= min_valid_acc:
                        if valid_dict["Lx_dist_true"] <= min_lx_dist:
                            print("Find model at bound step {} has smaller valid acc: {}".format(FLAGS.PIXEL_BOUND, valid_dict["adv_acc"]))
                            model.tf_save(sess, name=ckpt_name) # extra store
                            model.tf_save(sess)
                            min_valid_acc = valid_dict["adv_acc"]
                            min_lx_dist = valid_dict["Lx_dist_true"]
                            print("Trained params have been saved to '%s'" % FLAGS.AE_PATH)
                    
                print("******************************************************************")
            print()
            print()

             
            # Update Pixel bound
            if FLAGS.PIXEL_BOUND >= FLAGS.MIN_BOUND and FLAGS.PIXEL_BOUND <= FLAGS.MAX_BOUND and (epoch+1) % FLAGS.BOUND_CHANGE_EPOCHS == 0:
                if FLAGS.IS_TARGETED_ATTACK:
                    curr_valid_acc = max_valid_acc # max valid acc for current bound
                else:
                    curr_valid_acc = min_valid_acc # min valid acc for current bound
                
                # re-init
                sess.run(tf.global_variables_initializer())
                # Load target classifier
                model._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
                model.tf_load(sess) # corresponding model for min valid acc
                model.tf_load(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")

                if FLAGS.USE_IMITATION == False:
                    postfix = "_no_imit"
                elif FLAGS.ONLY_IMITATION:
                    postfix = "_only_imit"
                elif FLAGS.USE_DISTANCE == False:
                    postfix = "_no_dist"
                elif FLAGS.ONLY_DISTANCE:
                    postfix = "_only_dist"
                elif FLAGS.USE_MISCLASSIFY == False:
                    postfix = "_no_misc"
                elif FLAGS.ONLY_MISCLASSIFY:
                    postfix = "_only_misc"
                elif FLAGS.USE_ATT == False:
                    postfix = "_no_att"
                elif FLAGS.NUM_ENC_RES_BLOCK == 0 and FLAGS.ENC_RES_BLOCK_SIZE == 0:
                    postfix = "_no_res"
                elif FLAGS.LABEL_CONDITIONING == False:
                    postfix = "_no_lc"
                else: 
                    postfix = ""
                if FLAGS.IS_TARGETED_ATTACK:
                    valid_log_name = "tgt_tiny_valid_log{}.txt".format(postfix)
                else:
                    valid_log_name = "untgt_tiny_valid_log{}.txt".format(postfix)
                valid_dict = test_info(sess, model, None, graph_dict, valid_log_name, total_batch=total_valid_batch, valid=True)
                print()
                print("Changing bounds...")
                print("Roll back hit: {}".format(roll_back_hit))
                print("Validation accuracy of the current bound step: {}".format(curr_valid_acc))
                print("Validation accuracy of the previous bound step: {}".format(prev_valid_acc))
                print("Validation accuracy of the model restored: {}".format(valid_dict["adv_acc"]))
                

                if FLAGS.IS_TARGETED_ATTACK:
                    absolute_diff = prev_valid_acc - curr_valid_acc
                else:
                    absolute_diff = curr_valid_acc - prev_valid_acc
                if curr_valid_acc != 0:
                    acc_change_ratio =  absolute_diff / curr_valid_acc
                else:
                    acc_change_ratio = 0.0
                # FLAGS.ABS_DIFF_THRESHOLD 5e-4
                # change_itr != 0 : not first time
                if change_itr != 0 and roll_back_hit < FLAGS.ROLL_BACK_THRESHOLD and absolute_diff > FLAGS.ABS_DIFF_THRESHOLD and acc_change_ratio > FLAGS.ADAPTIVE_UP_THRESHOLD: # roll back
                    # re-init
                    sess.run(tf.global_variables_initializer())
                    # Load target classifier
                    model._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
                    model.tf_load(sess, name='deep_cae_last.ckpt')
                    model.tf_load(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")
                    
                    if FLAGS.IS_TARGETED_ATTACK:
                        max_valid_acc = last_max_valid_acc
                    else:
                        min_valid_acc = last_min_valid_acc
                    min_lx_dist = last_min_lx_dist
                    # reset
                    prev_valid_acc = last_prev_valid_acc
                    change_itr = last_change_itr
                    FLAGS.PIXEL_BOUND = last_pixel_bound
                    FLAGS.BETA_X_TRUE = last_beta_x_true
                    FLAGS.BETA_X_FAKE = last_beta_x_fake
                    FLAGS.BETA_Y_TRANS = last_beta_y_trans
                    FLAGS.BETA_Y_FAKE = last_beta_y_fake
                    FLAGS.BETA_Y_FAKE2 = last_beta_y_fake2
                    FLAGS.BETA_Y_CLEAN = last_beta_y_clean
                    # update bound change rate
                    if FLAGS.BOUND_CHANGE_RATE < INIT_BOUND_CHANGE_RATE:
                        FLAGS.BOUND_CHANGE_RATE = INIT_BOUND_CHANGE_RATE
                    FLAGS.BOUND_CHANGE_RATE = FLAGS.BOUND_CHANGE_RATE * FLAGS.ADAPTIVE_BOUND_INC_RATE
                    # update roll back
                    if last_pixel_bound == roll_back_pixel:
                        roll_back_hit += 1
                    else:
                        roll_back_pixel = last_pixel_bound
                        roll_back_hit = 0
                    # save model
                    model.tf_save(sess)
                else:
                    if roll_back_hit == 0 and acc_change_ratio < FLAGS.ADAPTIVE_LOW_THRESHOLD: # sppedup
                        FLAGS.BOUND_CHANGE_RATE = FLAGS.BOUND_CHANGE_RATE * FLAGS.ADAPTIVE_BOUND_DEC_RATE
                    if roll_back_pixel != FLAGS.PIXEL_BOUND and roll_back_hit >= FLAGS.ROLL_BACK_THRESHOLD:
                        roll_back_hit = 0
                    # save current for following use
                    model.tf_save(sess, name='deep_cae_last.ckpt')
                    if FLAGS.IS_TARGETED_ATTACK:
                        last_max_valid_acc = max_valid_acc
                    else:
                        last_min_valid_acc = min_valid_acc
                    last_min_lx_dist = min_lx_dist
                    # save from previous
                    last_change_itr = change_itr
                    last_prev_valid_acc = prev_valid_acc
                    last_pixel_bound = FLAGS.PIXEL_BOUND
                    last_beta_x_true = FLAGS.BETA_X_TRUE
                    last_beta_x_fake = FLAGS.BETA_X_FAKE
                    last_beta_y_trans = FLAGS.BETA_Y_TRANS
                    last_beta_y_fake = FLAGS.BETA_Y_FAKE
                    last_beta_y_fake2 = FLAGS.BETA_Y_FAKE2
                    last_beta_y_clean = FLAGS.BETA_Y_CLEAN
                    # update for following use
                    change_itr += 1
                    prev_valid_acc = curr_valid_acc
                    if FLAGS.IS_TARGETED_ATTACK:
                        max_valid_acc = 0.0
                    else:
                        min_valid_acc = 1.0
                    min_lx_dist = np.inf
                    FLAGS.PIXEL_BOUND = model_utils.change_coef(last_pixel_bound,  FLAGS.BOUND_CHANGE_RATE, change_itr,
                                                                FLAGS.BOUND_CHANGE_TYPE)
                    if last_pixel_bound - FLAGS.PIXEL_BOUND < 8e-5:
                        FLAGS.BOUND_CHANGE_RATE = INIT_BOUND_CHANGE_RATE
                    FLAGS.BETA_X_TRUE = INIT_BETA_X_TRUE * FLAGS.BETA_X_TRUE_CHANGE_RATE * math.ceil(INIT_PIXEL_BOUND / FLAGS.PIXEL_BOUND)
                    FLAGS.BETA_X_FAKE = INIT_BETA_X_FAKE * FLAGS.BETA_X_FAKE_CHANGE_RATE * math.ceil(INIT_PIXEL_BOUND / FLAGS.PIXEL_BOUND)
                    FLAGS.BETA_Y_TRANS = INIT_BETA_Y_TRANS * FLAGS.BETA_Y_TRANS_CHANGE_RATE * math.ceil(INIT_PIXEL_BOUND / FLAGS.PIXEL_BOUND)
                    FLAGS.BETA_Y_FAKE = INIT_BETA_Y_FAKE * FLAGS.BETA_Y_FAKE_CHANGE_RATE * math.ceil(INIT_PIXEL_BOUND / FLAGS.PIXEL_BOUND)
                    FLAGS.BETA_Y_FAKE2 = INIT_BETA_Y_FAKE2 * FLAGS.BETA_Y_FAKE2_CHANGE_RATE * math.ceil(INIT_PIXEL_BOUND / FLAGS.PIXEL_BOUND)
                    FLAGS.BETA_Y_CLEAN = INIT_BETA_Y_CLEAN * FLAGS.BETA_Y_CLEAN_CHANGE_RATE * math.ceil(INIT_PIXEL_BOUND / FLAGS.PIXEL_BOUND)               
                    
                # reset learning rate
                sess.run(fetches=[model_lr_reset_op])

            
        print("Optimization Finished!")


        #train_writer.close() 
        #valid_writer.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tf.app.run()
