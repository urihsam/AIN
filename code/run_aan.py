import aan
import os
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from dependency import *
from utils import model_utils
from utils.data_utils import dataset


model_utils.set_flags()

data = dataset(FLAGS.DATA_DIR, normalize=FLAGS.NORMALIZE, biased=FLAGS.BIASED)


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test_info(sess, model, test_writer, graph_dict, total_batch=None, valid=False):
    model_loss_x, (model_lx_true, model_lx_fake), (model_lx_dist_true, model_lx_dist_fake), \
            (model_max_dist_true, model_max_dist_fake) = model.loss_x(graph_dict["beta_x_t_holder"], graph_dict["beta_x_f_holder"])
    model_loss_y, (model_ly_trans, model_ly_fake, model_ly_clean), (model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean) =\
        model.loss_y(graph_dict["beta_y_t_holder"], graph_dict["beta_y_f_holder"], graph_dict["beta_y_c_holder"],
                     graph_dict["kappa_t_holder"], graph_dict["kappa_f_holder"], graph_dict["kappa_c_holder"]
                    )
    model_loss, model_recon_loss, model_sparse_loss, model_var_loss, model_reg = \
        model.loss(graph_dict["partial_loss_holder"], model_loss_x, model_loss_y)
    fetches = [model._target_accuracy, model._target_adv_accuracy, model._target_fake_accuracy, 
            model_loss, model_recon_loss, model_sparse_loss, model_var_loss, model_reg, 
            model_loss_x, model_lx_true, model_lx_fake, 
            model_lx_dist_true, model_lx_dist_fake,
            model_max_dist_true, model_max_dist_fake,
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
    loss = 0; recon_loss = 0; sparse_loss = 0; var_loss = 0; reg = 0;
    l_x = 0; Lx_true = 0; Lx_fake = 0; Lx_dist_true = 0; Lx_dist_fake = 0;
    max_dist_true = 0; max_dist_fake = 0;
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
            graph_dict["kappa_t_holder"]: FLAGS.KAPPA_FOR_TRANS,
            graph_dict["kappa_f_holder"]: FLAGS.KAPPA_FOR_FAKE,
            graph_dict["kappa_c_holder"]: FLAGS.KAPPA_FOR_CLEAN,
            graph_dict["partial_loss_holder"]: FLAGS.PARTIAL_LOSS,
            graph_dict["is_training"]: False
        }
        
        batch_acc, batch_adv_acc, batch_fake_acc, batch_loss, batch_recon_loss, batch_sparse_loss, batch_var_loss, batch_reg, \
            batch_l_x, batch_lx_true, batch_lx_fake, batch_Lx_dist_true, batch_Lx_dist_fake, \
            batch_max_dist_true, batch_max_dist_fake, \
            batch_l_y, batch_Ly_trans, batch_Ly_fake, batch_Ly_clean,\
            batch_Ly_dist_trans, batch_Ly_dist_fake, batch_Ly_dist_clean, \
            summary = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_writer.add_summary(summary, idx)
        acc += batch_acc
        adv_acc += batch_adv_acc
        fake_acc += batch_fake_acc
        loss += batch_loss
        recon_loss += batch_recon_loss
        sparse_loss += batch_sparse_loss
        var_loss += batch_var_loss
        reg += batch_reg
        l_x += batch_l_x
        Lx_true += batch_lx_true
        Lx_fake += batch_lx_fake
        Lx_dist_true += batch_Lx_dist_true
        Lx_dist_fake += batch_Lx_dist_fake
        max_dist_true += batch_max_dist_true
        max_dist_fake += batch_max_dist_fake
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
    sparse_loss /= total_batch
    var_loss /= total_batch
    reg /= total_batch
    l_x /= total_batch
    Lx_true /= total_batch
    Lx_fake /= total_batch
    Lx_dist_true /= total_batch
    Lx_dist_fake /= total_batch
    max_dist_true /= total_batch
    max_dist_fake /= total_batch
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
    print("Loss = {:.4f}  Loss recon = {:.4f}  Loss sparse = {:.4f}  Loss var = {:.4f}  Loss reg = {:.4f}".format(
        loss, recon_loss, sparse_loss, var_loss, reg))
    print("Loss x = {:.4f}  Loss y = {:.4f}".format(l_x, l_y))
    print("Loss x for true = {:.4f} Loss x for fake = {:.4}".format(Lx_true, Lx_fake))
    print("Loss y for trans = {:.4f} Loss y for fake = {:.4f} Loss y for clean = {:.4f}".format(Ly_trans, Ly_fake, Ly_clean))
    print("Lx distance for true = {:.4f} Lx distance for fake = {:.4f}".format(Lx_dist_true, Lx_dist_fake))
    print("Max pixel distance for true = {:.4f} Max pixel distance for fake = {:.4f}".format(max_dist_true, max_dist_fake))
    print("Ly distance for trans = {:.4f} Ly distance for fake = {:.4f} Ly distance for clean = {:.4f}".format(
        Ly_dist_trans, Ly_dist_fake, Ly_dist_clean))
    res_dict = {"acc": acc, 
                "fake_acc": fake_acc,
                "adv_acc": adv_acc, 
                "loss": loss, 
                "sparse_loss": sparse_loss,
                "var_loss": var_loss,
                "l_x": l_x,
                "Lx_true": Lx_true,
                "Lx_fake": Lx_fake,
                "Lx_dist_true": Lx_dist_true,
                "Lx_dist_fake": Lx_dist_fake,
                "max_dist_true": max_dist_true,
                "max_dist_fake": max_dist_fake,
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
        kappa_t_holder = tf.placeholder(tf.float32, ())
        kappa_f_holder = tf.placeholder(tf.float32, ())
        kappa_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.string, ())
        is_training = tf.placeholder(tf.bool, ())

        model = aan.AAN(images_holder, label_holder, low_bound_holder, up_bound_holder, epsilon_holder, is_training)
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
        graph_dict["kappa_t_holder"] = kappa_t_holder
        graph_dict["kappa_f_holder"] = kappa_f_holder
        graph_dict["kappa_c_holder"] = kappa_c_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary
        

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        model.tf_load(sess)
        # tensorboard writer
        test_writer = model_utils.init_writer(FLAGS.TEST_LOG_PATH, g)
        print("\nTest")
        if FLAGS.local:
            total_test_batch = 2
        else:
            total_test_batch = None
        test_info(sess, model, test_writer, graph_dict, total_batch=total_test_batch)
        test_writer.close() 
        
        batch_xs, batch_ys = data.next_test_batch(FLAGS.BATCH_SIZE)
        feed_dict = {
            images_holder: batch_xs,
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
            im1 = Image.fromarray(np.uint8(batch_xs[i]))
            im2 = Image.fromarray(np.uint8(adv_images[i]))
            new_im.paste(im1, (x_offset, 0))
            new_im.paste(im2, (x_offset, y_offset))
            x_offset += im1.size[0]

        new_im.show()
        new_im.save('AAN_results.jpg')


def train():
    """
    """
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
        kappa_t_holder = tf.placeholder(tf.float32, ())
        kappa_f_holder = tf.placeholder(tf.float32, ())
        kappa_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.string, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = aan.AAN(images_holder, label_holder, low_bound_holder, up_bound_holder, epsilon_holder, is_training)
        model_loss_x, (model_lx_true, model_lx_fake), (model_lx_dist_true, model_lx_dist_fake), \
            (model_max_dist_true, model_max_dist_fake) = model.loss_x(beta_x_t_holder, beta_x_f_holder)
        model_loss_y, (model_ly_trans, model_ly_fake, model_ly_clean), (model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean) =\
            model.loss_y(beta_y_t_holder, beta_y_f_holder, beta_y_c_holder,
                         kappa_t_holder, kappa_f_holder, kappa_c_holder
                        )
        model_loss, model_recon_loss, model_sparse_loss, model_var_loss, model_reg = \
            model.loss(partial_loss_holder, model_loss_x, model_loss_y)
        model_optimization = model.optimization(model_loss)
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
        graph_dict["kappa_t_holder"] = kappa_t_holder
        graph_dict["kappa_f_holder"] = kappa_f_holder
        graph_dict["kappa_c_holder"] = kappa_c_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # Load target classifier
        model._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
        if FLAGS.load_AE:
            print("Autoencoder loaded.")
            model._autoencoder.tf_load(sess, FLAGS.AE_PATH)
        # For tensorboard
        train_writer = model_utils.init_writer(FLAGS.TRAIN_LOG_PATH, g)
        valid_writer = model_utils.init_writer(FLAGS.VALID_LOG_PATH, g)
        
        if FLAGS.local:
            total_train_batch = 2
            total_valid_batch = 2
        else:
            total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
            total_valid_batch = None
        
        min_adv_acc = np.Inf
        alert_count = 0
        
        for epoch in range(FLAGS.NUM_EPOCHS):
            start_time = time.time()
            for train_idx in range(total_train_batch):
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
                    kappa_t_holder: FLAGS.KAPPA_FOR_TRANS,
                    kappa_f_holder: FLAGS.KAPPA_FOR_FAKE,
                    kappa_c_holder: FLAGS.KAPPA_FOR_CLEAN,
                    is_training: True,
                    partial_loss_holder: FLAGS.PARTIAL_LOSS
                }
                """[res0, res1, res2] = sess.run([model.adv, model.fake, model.cross_entropy],
                                        feed_dict=feed_dict)
                import pdb; pdb.set_trace()"""
                # optimization
                fetches = [model_optimization, model_loss, 
                        model_recon_loss, model_sparse_loss, model_var_loss, model_reg, 
                        model_loss_x, model_lx_true, model_lx_fake, 
                        model_lx_dist_true, model_lx_dist_fake,
                        model_max_dist_true, model_max_dist_fake,
                        model_loss_y, model_ly_trans, model_ly_fake, model_ly_clean, 
                        model_ly_dist_trans, model_ly_dist_fake, model_ly_dist_clean,
                        merged_summary, model._target_fake_prediction]
                _, loss, recon_loss, sparse_loss, var_loss, reg, l_x, Lx_true, Lx_fake, Lx_dist_true, Lx_dist_fake, \
                    max_dist_true, max_dist_fake, l_y, Ly_trans, Ly_fake, Ly_clean, \
                    Ly_dist_trans, Ly_dist_fake, Ly_dist_clean, \
                    summary, fake_prediction = sess.run(fetches=fetches, feed_dict=feed_dict)
                
                #import pdb; pdb.set_trace()
                train_writer.add_summary(summary, train_idx)
                # Print info
                if train_idx % FLAGS.EVAL_FREQUENCY == (FLAGS.EVAL_FREQUENCY - 1):
                    print("Epoch: {}".format(epoch+1))
                    print("Hyper-params info:")
                    print("Using Partial Loss:", FLAGS.PARTIAL_LOSS)
                    print("Pixel bound: [{:.4f}, {:.4f}]  Epsilon: {:.4f}".format(
                        -1.0*FLAGS.PIXEL_BOUND, FLAGS.PIXEL_BOUND, FLAGS.EPSILON))
                    print("BETA_X_TRUE: {:.4f}  BETA_X_FAKE: {:.4f}   Beta_Y_TRANS: {:.4f}  Beta_Y_FAKE: {:.4f}  Beta_Y_CLEAN: {:.4f}".format(
                        FLAGS.BETA_X_TRUE, FLAGS.BETA_X_FAKE, FLAGS.BETA_Y_TRANS, FLAGS.BETA_Y_FAKE, FLAGS.BETA_Y_CLEAN
                    ))
                    print("Result:")
                    print("Loss = {:.4f}  Loss recon = {:.4f}  Loss sparse = {:.4f}  Loss var = {:.4f}  Loss reg = {:.4f}".format(
                        loss, recon_loss, sparse_loss, var_loss, reg))
                    print("Loss x = {:.4f}  Loss y = {:.4f}".format(l_x, l_y))
                    print("Loss x for true = {:.4f} Loss x for fake = {:.4}".format(Lx_true, Lx_fake))
                    print("Loss y for trans = {:.4f} Loss y for fake = {:.4f} Loss y for clean = {:.4f}".format(Ly_trans, Ly_fake, Ly_clean))
                    print("Lx distance for true = {:.4f} Lx distance for fake = {:.4f}".format(Lx_dist_true, Lx_dist_fake))
                    print("Max pixel distance for true = {:.4f} Max pixel distance for fake = {:.4f}".format(max_dist_true, max_dist_fake))
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
            if FLAGS.PARTIAL_LOSS != "FULL_LOSS":
                #Update loss x threshold
                if FLAGS.LOSS_X_THRESHOLD >= FLAGS.MIN_LOSS_X_THRE and FLAGS.LOSS_X_THRESHOLD <= FLAGS.MAX_LOSS_X_THRE and (epoch+1) % FLAGS.LOSS_X_THRE_CHANGE_EPOCHS == 0:
                    FLAGS.LOSS_X_THRESHOLD = FLAGS.LOSS_X_THRESHOLD * FLAGS.LOSS_X_THRE_CHANGE_RATE
            # Update Pixel bound
            if FLAGS.PIXEL_BOUND >= FLAGS.MIN_BOUND and FLAGS.PIXEL_BOUND <= FLAGS.MAX_BOUND and (epoch+1) % FLAGS.BOUND_CHANGE_EPOCHS == 0:
                FLAGS.PIXEL_BOUND = FLAGS.PIXEL_BOUND * FLAGS.BOUND_CHANGE_RATE
            # Update BETA_X_TRUE
            if FLAGS.BETA_X_TRUE >= FLAGS.MIN_BETA_X_TRUE and FLAGS.BETA_X_TRUE <= FLAGS.MAX_BETA_X_TRUE and (epoch+1) % FLAGS.BETA_X_TRUE_CHANGE_EPOCHS == 0:
                FLAGS.BETA_X_TRUE =  FLAGS.BETA_X_TRUE * FLAGS.BETA_X_TRUE_CHANGE_RATE
            # Update Beta_X_FAKE
            if FLAGS.BETA_X_FAKE >= FLAGS.MIN_BETA_X_FAKE and FLAGS.BETA_X_FAKE <= FLAGS.MAX_BETA_X_FAKE and (epoch+1) % FLAGS.BETA_X_FAKE_CHANGE_EPOCHS == 0:
                FLAGS.BETA_X_FAKE =  FLAGS.BETA_X_FAKE * FLAGS.BETA_X_FAKE_CHANGE_RATE
            # Update BETA_Y_TRANS
            if FLAGS.BETA_Y_TRANS >= FLAGS.MIN_BETA_Y_TRANS and FLAGS.BETA_Y_TRANS <= FLAGS.MAX_BETA_Y_TRANS and (epoch+1) % FLAGS.BETA_Y_TRANS_CHANGE_EPOCHS == 0:
                FLAGS.BETA_Y_TRANS =  FLAGS.BETA_Y_TRANS * FLAGS.BETA_Y_TRANS_CHANGE_RATE
            # Update Beta_Y_CLEAN
            if FLAGS.BETA_Y_CLEAN >= FLAGS.MIN_BETA_Y_CLEAN and FLAGS.BETA_Y_CLEAN <= FLAGS.MAX_BETA_Y_CLEAN and (epoch+1) % FLAGS.BETA_Y_CLEAN_CHANGE_EPOCHS == 0:
                FLAGS.BETA_Y_CLEAN =  FLAGS.BETA_Y_CLEAN * FLAGS.BETA_Y_CLEAN_CHANGE_RATE
            
                
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
            # validation
            print("\n******************************************************************")
            print("Validation")
            valid_dict = test_info(sess, model, valid_writer, graph_dict, total_batch=total_valid_batch, valid=True)
            
            if valid_dict["adv_acc"] < min_adv_acc:
                min_adv_acc = valid_dict["adv_acc"]
                alert_count = 0
            else:
                alert_count += 1
            # early stopping
            if FLAGS.early_stopping:
                if alert_count >= FLAGS.EARLY_STOPPING_THRESHOLD:
                    print("Early Stopped.")
                    break
            if alert_count >= FLAGS.MODIFY_KAPPA_THRESHOLD and valid_dict["adv_acc"] >= 2.0 * valid_dict["fake_acc"]:
                # reset
                min_adv_acc = np.Inf
                alert_count = 0
                FLAGS.KAPPA_FOR_TRANS = FLAGS.KAPPA_TRANS_CHANGE_RATE * FLAGS.KAPPA_FOR_TRANS
                FLAGS.KAPPA_FOR_CLEAN = FLAGS.KAPPA_CLEAN_CHANGE_RATE * FLAGS.KAPPA_FOR_CLEAN
                print("Kappa trans was changed to {:.4f}".format(FLAGS.KAPPA_FOR_TRANS))
                print("Kappa clean was changed to {:.4f}".format(FLAGS.KAPPA_FOR_CLEAN))
            
            print("******************************************************************")
            print()
            print()
        print("Optimization Finished!")

        model.tf_save(sess)
        print("Trained params have been saved to '%s'" % FLAGS.AE_PATH)

        train_writer.close() 
        valid_writer.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
