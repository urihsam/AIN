import nn.attain_v4_imagenet as attain
import os
from PIL import Image
from dependency import *
import utils.model_utils_imagenet as model_utils
from utils.data_utils_imagenet import dataset


model_utils.set_flags()

data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)


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
        model.loss_y(graph_dict["beta_y_t_holder"], graph_dict["beta_y_f_holder"], graph_dict["beta_y_c_holder"]
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
    
    for idx in range(total_batch):
        if valid:
            batch_xs, batch_ys = data.next_valid_batch(FLAGS.BATCH_SIZE)
        else:
            batch_xs, batch_ys = data.next_test_batch(FLAGS.BATCH_SIZE)

        feed_dict = {
            graph_dict["images_holder"]: batch_xs,
            graph_dict["label_holder"]: batch_ys,
            graph_dict["low_bound_holder"]: -1.0*FLAGS.PIXEL_BOUND,
            graph_dict["up_bound_holder"]: 1.0*FLAGS.PIXEL_BOUND,
            graph_dict["epsilon_holder"]: FLAGS.EPSILON,
            graph_dict["beta_x_t_holder"]: FLAGS.BETA_X_TRUE,
            graph_dict["beta_x_f_holder"]: FLAGS.BETA_X_FAKE,
            graph_dict["beta_y_t_holder"]: FLAGS.BETA_Y_TRANS,
            graph_dict["beta_y_f_holder"]: FLAGS.BETA_Y_FAKE,
            graph_dict["beta_y_c_holder"]: FLAGS.BETA_Y_CLEAN,
            graph_dict["partial_loss_holder"]: FLAGS.PARTIAL_LOSS,
            graph_dict["is_training"]: False
        }
        
        batch_acc, batch_adv_acc, batch_fake_acc, batch_loss, batch_recon_loss, batch_label_loss, batch_sparse_loss, batch_var_loss, batch_reg, \
            batch_l_x, batch_lx_true, batch_lx_fake, batch_Lx_dist_true, batch_Lx_dist_fake, batch_Lx_dist_trans,\
            batch_max_dist_true, batch_max_dist_fake, batch_max_dist_trans,\
            batch_l_y, batch_Ly_trans, batch_Ly_fake, batch_Ly_clean,\
            batch_Ly_dist_trans, batch_Ly_dist_fake, batch_Ly_dist_clean, \
            summary = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_writer.add_summary(summary, idx)
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

    #adv_images = X+adv_noises
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
        file.write("############################################")
    
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
        low_bound_holder = tf.placeholder(tf.float32, ())
        up_bound_holder = tf.placeholder(tf.float32, ())
        epsilon_holder = tf.placeholder(tf.float32, ())
        beta_x_t_holder = tf.placeholder(tf.float32, ())
        beta_x_f_holder = tf.placeholder(tf.float32, ())
        beta_y_t_holder = tf.placeholder(tf.float32, ())
        beta_y_f_holder = tf.placeholder(tf.float32, ())
        beta_y_c_holder = tf.placeholder(tf.float32, ())
        #kappa_t_holder = tf.placeholder(tf.float32, ())
        #kappa_f_holder = tf.placeholder(tf.float32, ())
        #kappa_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.string, ())
        is_training = tf.placeholder(tf.bool, ())

        model = attain.ATTAIN(images_holder, label_holder, low_bound_holder, up_bound_holder, epsilon_holder, is_training)
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["low_bound_holder"] = low_bound_holder
        graph_dict["up_bound_holder"] = up_bound_holder
        graph_dict["epsilon_holder"] = epsilon_holder
        graph_dict["beta_x_t_holder"] = beta_x_t_holder
        graph_dict["beta_x_f_holder"] = beta_x_f_holder
        graph_dict["beta_y_t_holder"] = beta_y_t_holder
        graph_dict["beta_y_f_holder"] = beta_y_f_holder
        graph_dict["beta_y_c_holder"] = beta_y_c_holder
        #graph_dict["kappa_t_holder"] = kappa_t_holder
        #graph_dict["kappa_f_holder"] = kappa_f_holder
        #graph_dict["kappa_c_holder"] = kappa_c_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary
        

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # Load target classifier
        model._target.tf_load(sess, FLAGS.RESNET50_PATH, 'resnet_v2_50.ckpt')
        model.tf_load(sess)
        model.tf_load(sess, scope=FLAGS.LBL_NAME, name="label_states.ckpt")
        # tensorboard writer
        test_writer = model_utils.init_writer(FLAGS.TEST_LOG_PATH, g)
        print("\nTest")
        if FLAGS.local:
            total_test_batch = 2
        else:
            total_test_batch = None
        test_info(sess, model, test_writer, graph_dict, "test_log.txt", total_batch=total_test_batch)
        test_writer.close() 
        
        batch_xs, batch_ys = data.next_test_batch(FLAGS.BATCH_SIZE)
        feed_dict = {
            images_holder: batch_xs,
            label_holder: batch_ys,
            low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
            up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
            is_training: False
        }
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
        new_im.save('AIN_results.jpg')


def train():
    """
    """
    INIT_PIXEL_BOUND = FLAGS.PIXEL_BOUND
    INIT_BETA_X_TRUE = FLAGS.BETA_X_TRUE
    INIT_BETA_X_FAKE = FLAGS.BETA_X_FAKE
    INIT_BETA_Y_TRANS = FLAGS.BETA_Y_TRANS
    INIT_BETA_Y_CLEAN = FLAGS.BETA_Y_CLEAN
    import time
    tf.reset_default_graph()
    g = tf.get_default_graph()
    # attack_target = 8
    with g.as_default():
        # Placeholder nodes.
        images_holder = tf.placeholder(tf.float32, [None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS])
        label_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        low_bound_holder = tf.placeholder(tf.float32, ())
        up_bound_holder = tf.placeholder(tf.float32, ())
        epsilon_holder = tf.placeholder(tf.float32, ())
        beta_x_t_holder = tf.placeholder(tf.float32, ())
        beta_x_f_holder = tf.placeholder(tf.float32, ())
        beta_y_t_holder = tf.placeholder(tf.float32, ())
        beta_y_f_holder = tf.placeholder(tf.float32, ())
        beta_y_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.string, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = attain.ATTAIN(images_holder, label_holder, low_bound_holder, up_bound_holder, epsilon_holder, is_training)

        # pre-training
        pre_label_loss = FLAGS.GAMMA_PRE_L * model.pre_loss_label
        pre_op, _, _, _ = model.optimization(pre_label_loss, scope="PRE_OPT")
        # model training
        
        model_loss_x, (model_lx_true, model_lx_fake), (model_lx_dist_true, model_lx_dist_fake, model_lx_dist_trans), \
            (model_max_dist_true, model_max_dist_fake, model_max_dist_trans) = model.loss_x(beta_x_t_holder, beta_x_f_holder)
        model_loss_y, (model_ly_trans, model_ly_fake, model_ly_clean), (model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean) =\
            model.loss_y(beta_y_t_holder, beta_y_f_holder, beta_y_c_holder
                        )
        model_loss, model_recon_loss, model_label_loss, model_sparse_loss, model_var_loss, model_reg = \
            model.loss(partial_loss_holder, model_loss_x, model_loss_y)
        model_op, (model_zero_op,  model_accum_op, model_avg_op), model_lr_reset_op, model_lr = model.optimization(model_loss, accum_iters=FLAGS.NUM_ACCUM_ITERS)
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["low_bound_holder"] = low_bound_holder
        graph_dict["up_bound_holder"] = up_bound_holder
        graph_dict["epsilon_holder"] = epsilon_holder
        graph_dict["beta_x_t_holder"] = beta_x_t_holder
        graph_dict["beta_x_f_holder"] = beta_x_f_holder
        graph_dict["beta_y_t_holder"] = beta_y_t_holder
        graph_dict["beta_y_f_holder"] = beta_y_f_holder
        graph_dict["beta_y_c_holder"] = beta_y_c_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary

    with tf.Session(graph=g) as sess:
        #import pdb; pdb.set_trace()
        sess.run(tf.global_variables_initializer())
        # Load target classifier
        #import pdb; pdb.set_trace()
        model._target.tf_load(sess, FLAGS.RESNET50_PATH, 'resnet_v2_50.ckpt')
        if FLAGS.load_AE:
            print("Autoencoder loaded.")
            model.tf_load(sess)
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
        
        min_adv_acc = np.Inf
        alert_count = 0

        if FLAGS.train_label:
            print("Pre-training...")
            for epoch in range(FLAGS.NUM_PRE_EPOCHS):
                start_time = time.time()
                for train_idx in range(total_pre_train_batch):
                    batch_xs, batch_ys = data.next_train_batch(FLAGS.BATCH_SIZE)
                    feed_dict = {
                        label_holder: batch_ys,
                        is_training: True
                    }
                    fetches = [pre_op, pre_label_loss]
                    _, pre_loss = sess.run(fetches=fetches, feed_dict=feed_dict)
                    if train_idx % 10000 == 0:
                        print ("train idx: {}".format(train_idx))

                if epoch % FLAGS.PRE_EVAL_FREQUENCY == (FLAGS.PRE_EVAL_FREQUENCY - 1):
                    print("Training Result:")
                    print("Pre Loss label = {:.4f}".format(pre_loss))
                    pre_valid_loss = 0
                    for valid_idx in range(total_pre_valid_batch):
                        batch_xs, batch_ys = data.next_valid_batch(FLAGS.BATCH_SIZE)
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
        for epoch in range(FLAGS.NUM_EPOCHS):
            start_time = time.time()
            for train_idx in range(total_train_batch):
                if FLAGS.NUM_ACCUM_ITERS != 1:
                    for accum_idx in range(FLAGS.NUM_ACCUM_ITERS):
                        batch_xs, batch_ys = data.next_train_batch(FLAGS.BATCH_SIZE)

                        feed_dict = {
                            images_holder: batch_xs,
                            label_holder: batch_ys,
                            low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
                            up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
                            epsilon_holder: FLAGS.EPSILON,
                            beta_x_t_holder: FLAGS.BETA_X_TRUE,
                            beta_x_f_holder: FLAGS.BETA_X_FAKE,
                            beta_y_t_holder: FLAGS.BETA_Y_TRANS,
                            beta_y_f_holder: FLAGS.BETA_Y_FAKE,
                            beta_y_c_holder: FLAGS.BETA_Y_CLEAN,
                            #kappa_t_holder: FLAGS.KAPPA_FOR_TRANS,
                            #kappa_f_holder: FLAGS.KAPPA_FOR_FAKE,
                            #kappa_c_holder: FLAGS.KAPPA_FOR_CLEAN,
                            is_training: True,
                            partial_loss_holder: FLAGS.PARTIAL_LOSS
                        }
                        sess.run(fetches=[model_accum_op], feed_dict=feed_dict)
                    sess.run(fetches=[model_avg_op])
                
                else:
                    batch_xs, batch_ys = data.next_train_batch(FLAGS.BATCH_SIZE)
                    feed_dict = {
                        images_holder: batch_xs,
                        label_holder: batch_ys,
                        low_bound_holder: -1.0*FLAGS.PIXEL_BOUND,
                        up_bound_holder: 1.0*FLAGS.PIXEL_BOUND,
                        epsilon_holder: FLAGS.EPSILON,
                        beta_x_t_holder: FLAGS.BETA_X_TRUE,
                        beta_x_f_holder: FLAGS.BETA_X_FAKE,
                        beta_y_t_holder: FLAGS.BETA_Y_TRANS,
                        beta_y_f_holder: FLAGS.BETA_Y_FAKE,
                        beta_y_c_holder: FLAGS.BETA_Y_CLEAN,
                        #kappa_t_holder: FLAGS.KAPPA_FOR_TRANS,
                        #kappa_f_holder: FLAGS.KAPPA_FOR_FAKE,
                        #kappa_c_holder: FLAGS.KAPPA_FOR_CLEAN,
                        is_training: True,
                        partial_loss_holder: FLAGS.PARTIAL_LOSS
                    }
                
                #import pdb; pdb.set_trace()
                #res = sess.run([model._target_prediction, model.label], feed_dict)
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
                train_writer.add_summary(summary, train_idx)
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
                    print("Lx distance for true = {:.4f} Lx distance for fake = {:.4f} Lx distance for trans = {:.4f}".format(
                        Lx_dist_true, Lx_dist_fake, Lx_dist_trans))
                    print("Max pixel distance for true = {:.4f} Max pixel distance for fake = {:.4f} Max pixel distance for trans = {:.4f}".format(
                        max_dist_true, max_dist_fake, max_dist_trans))
                    print("Ly distance for trans = {:.4f} Ly distance for fake = {:.4f} Ly distance for clean = {:.4f}".format(
                        Ly_dist_trans, Ly_dist_fake, Ly_dist_clean))
                    print()
                    model.tf_save(sess) # save checkpoint
                    
                if FLAGS.PARTIAL_LOSS != "FULL_LOSS":
                    if train_idx % FLAGS.LOSS_CHANGE_FREQUENCY == (FLAGS.LOSS_CHANGE_FREQUENCY - 1):
                        if Lx_dist <= FLAGS.LOSS_X_THRESHOLD:
                            FLAGS.PARTIAL_LOSS = "LOSS_Y"
                        else:
                            FLAGS.PARTIAL_LOSS = "LOSS_X"
                      
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
            # validation
            print("\n******************************************************************")
            print("Validation")
            valid_dict = test_info(sess, model, valid_writer, graph_dict, "valid_log.txt", total_batch=total_valid_batch, valid=True)
            

            if valid_dict["adv_acc"] > valid_dict["fake_acc"] and valid_dict["adv_acc"] > 0.1:
                break
            else:
                model.tf_save(sess)
                print("Trained params have been saved to '%s'" % FLAGS.AE_PATH)
            
            print("******************************************************************")
            print()
            print()

             
            # Update Pixel bound
            if FLAGS.PIXEL_BOUND >= FLAGS.MIN_BOUND and FLAGS.PIXEL_BOUND <= FLAGS.MAX_BOUND and (epoch+1) % FLAGS.BOUND_CHANGE_EPOCHS == 0:
                FLAGS.PIXEL_BOUND = model_utils.change_coef(INIT_PIXEL_BOUND,  FLAGS.BOUND_CHANGE_RATE, (epoch+1) // FLAGS.BOUND_CHANGE_EPOCHS,
                                                            FLAGS.BOUND_CHANGE_TYPE)
                # reser learning rate
                sess.run(fetches=[model_lr_reset_op])
            
            # Update BETA_X_TRUE
            if FLAGS.BETA_X_TRUE >= FLAGS.MIN_BETA_X_TRUE and FLAGS.BETA_X_TRUE <= FLAGS.MAX_BETA_X_TRUE and (epoch+1) % FLAGS.BETA_X_TRUE_CHANGE_EPOCHS == 0:
                FLAGS.BETA_X_TRUE =  model_utils.change_coef(INIT_BETA_X_TRUE, FLAGS.BETA_X_TRUE_CHANGE_RATE, (epoch+1) // FLAGS.BETA_X_TRUE_CHANGE_EPOCHS,
                                                             FLAGS.BETA_X_TRUE_CHANGE_TYPE)
            # Update Beta_X_FAKE
            if FLAGS.BETA_X_FAKE >= FLAGS.MIN_BETA_X_FAKE and FLAGS.BETA_X_FAKE <= FLAGS.MAX_BETA_X_FAKE and (epoch+1) % FLAGS.BETA_X_FAKE_CHANGE_EPOCHS == 0:
                FLAGS.BETA_X_FAKE =  model_utils.change_coef(INIT_BETA_X_FAKE, FLAGS.BETA_X_FAKE_CHANGE_RATE, (epoch+1) // FLAGS.BETA_X_FAKE_CHANGE_EPOCHS,
                                                             FLAGS.BETA_X_FAKE_CHANGE_TYPE)
            # Update BETA_Y_TRANS
            if FLAGS.BETA_Y_TRANS >= FLAGS.MIN_BETA_Y_TRANS and FLAGS.BETA_Y_TRANS <= FLAGS.MAX_BETA_Y_TRANS and (epoch+1) % FLAGS.BETA_Y_TRANS_CHANGE_EPOCHS == 0:
                FLAGS.BETA_Y_TRANS =  model_utils.change_coef(INIT_BETA_Y_TRANS, FLAGS.BETA_Y_TRANS_CHANGE_RATE, (epoch+1) // FLAGS.BETA_Y_TRANS_CHANGE_EPOCHS,
                                                              FLAGS.BETA_Y_TRANS_CHANGE_TYPE)
            # Update Beta_Y_CLEAN
            if FLAGS.BETA_Y_CLEAN >= FLAGS.MIN_BETA_Y_CLEAN and FLAGS.BETA_Y_CLEAN <= FLAGS.MAX_BETA_Y_CLEAN and (epoch+1) % FLAGS.BETA_Y_CLEAN_CHANGE_EPOCHS == 0:
                FLAGS.BETA_Y_CLEAN =  model_utils.change_coef(INIT_BETA_Y_CLEAN, FLAGS.BETA_Y_CLEAN_CHANGE_RATE, (epoch+1) // FLAGS.BETA_Y_CLEAN_CHANGE_EPOCHS,
                                                              FLAGS.BETA_Y_CLEAN_CHANGE_TYPE)

            
        print("Optimization Finished!")


        train_writer.close() 
        valid_writer.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
