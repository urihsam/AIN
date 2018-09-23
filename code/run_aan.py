import aan
import os
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from dependency import *
from utils import model_utils
from utils.data_utils import dataset


model_utils.set_flags()

data = dataset(FLAGS.DATA_DIR, normalize=False)


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test_info(sess, model, test_writer, graph_dict, total_batch=None, valid=False):
    model_loss_x, model_lx_dist, model_max_dist, _ = model.loss_x(graph_dict["beta_x_holder"])
    model_loss_y, (model_ly_least, model_ly_fake, model_ly_clean), (model_ly_dist_least, model_ly_dist_fake, model_ly_dist_clean) =\
        model.loss_y(graph_dict["beta_y_l_holder"], graph_dict["beta_y_f_holder"], graph_dict["beta_y_c_holder"])
    model_loss = model.loss(graph_dict["partial_loss_holder"], model_loss_x, model_loss_y)
    fetches = [model._target_accuracy, model._target_adv_accuracy, model._target_fake_accuracy, model_loss, model_loss_x,
               model_lx_dist, model_max_dist, model_loss_y, model_ly_least, model_ly_fake, model_ly_clean, 
               model_ly_dist_least, model_ly_dist_fake, model_ly_dist_clean,
               graph_dict["merged_summary"]]
    if total_batch is None:
        if valid:
            total_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
        else:
            total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    else: total_batch = total_batch

    acc = 0; adv_acc = 0; fake_acc = 0; loss = 0; 
    l_x = 0; Lx_dist = 0; max_dist = 0; 
    l_y = 0; Ly_least = 0; Ly_fake = 0; Ly_clean = 0; 
    Ly_dist_least = 0; Ly_dist_fake = 0; Ly_dist_clean = 0
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
            graph_dict["beta_x_holder"]: FLAGS.BETA_X,
            graph_dict["beta_y_l_holder"]: FLAGS.BETA_Y_LEAST,
            graph_dict["beta_y_f_holder"]: FLAGS.BETA_Y_FAKE,
            graph_dict["beta_y_c_holder"]: FLAGS.BETA_Y_CLEAN,
            graph_dict["partial_loss_holder"]: FLAGS.PARTIAL_LOSS,
            graph_dict["is_training"]: False
        }
        
        batch_acc, batch_adv_acc, batch_fake_acc, batch_loss, batch_l_x, batch_Lx_dist, batch_max_dist, \
            batch_l_y, batch_Ly_least, batch_Ly_fake, batch_Ly_clean,\
            batch_Ly_dist_least, batch_Ly_dist_fake, batch_Ly_dist_clean, \
            summary = sess.run(fetches=fetches, feed_dict=feed_dict)
        test_writer.add_summary(summary, idx)
        acc += batch_acc
        adv_acc += batch_adv_acc
        fake_acc += batch_fake_acc
        loss += batch_loss
        l_x += batch_l_x
        Lx_dist += batch_Lx_dist
        max_dist += batch_max_dist
        l_y += batch_l_y
        Ly_least += batch_Ly_least
        Ly_fake += batch_Ly_fake
        Ly_clean += batch_Ly_clean
        Ly_dist_least += batch_Ly_dist_least
        Ly_dist_fake += batch_Ly_dist_fake
        Ly_dist_clean += batch_Ly_dist_clean
        
    acc /= total_batch
    adv_acc /= total_batch
    fake_acc /= total_batch
    loss /= total_batch
    l_x /= total_batch
    Lx_dist /= total_batch
    max_dist /= total_batch
    l_y /= total_batch
    Ly_least /= total_batch
    Ly_fake /= total_batch
    Ly_clean /= total_batch
    Ly_dist_least /= total_batch
    Ly_dist_fake /= total_batch
    Ly_dist_clean /= total_batch

    #adv_images = X+adv_noises
    print('Original accuracy: {0:0.5f}'.format(acc))
    print('Faked accuracy: {0:0.5f}'.format(fake_acc))
    print('Attacked accuracy: {0:0.5f}'.format(adv_acc))
    print("loss = {:.4f}  Loss x = {:.4f}  Loss y = {:.4f}".format(loss, l_x, l_y))
    print("Loss y for least = {:.4f} Loss y for fake = {:.4f} Loss y for clean = {:.4f}".format(Ly_least, Ly_fake, Ly_clean))
    print("Lx distance = {:.4f} Max pixel distance = {:.4f}".format(Lx_dist, max_dist))
    print("Ly distance for least = {:.4f} Ly distance for fake = {:.4f} Ly distance for clean = {:.4f}".format(
        Ly_dist_least, Ly_dist_fake, Ly_dist_clean))
    res_dict = {"acc": acc, 
                "adv_acc": adv_acc, 
                "loss": loss, 
                "l_x": l_x,
                "Lx_dist": Lx_dist,
                "max_dist": max_dist,
                "l_y": l_y, 
                "Ly_least": Ly_least,
                "Ly_fake": Ly_fake,
                "Ly_clean": Ly_clean,
                "Ly_dist_least": Ly_dist_least,
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
        beta_x_holder = tf.placeholder(tf.float32, ())
        beta_y_l_holder = tf.placeholder(tf.float32, ())
        beta_y_f_holder = tf.placeholder(tf.float32, ())
        beta_y_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.bool, ())
        is_training = tf.placeholder(tf.bool, ())

        model = aan.AAN(images_holder, label_holder, low_bound_holder, up_bound_holder, epsilon_holder, is_training)
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["low_bound_holder"] = low_bound_holder
        graph_dict["up_bound_holder"] = up_bound_holder
        graph_dict["epsilon_holder"] = epsilon_holder
        graph_dict["beta_x_holder"] = beta_x_holder
        graph_dict["beta_y_l_holder"] = beta_y_l_holder
        graph_dict["beta_y_f_holder"] = beta_y_f_holder
        graph_dict["beta_y_c_holder"] = beta_y_c_holder
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
        beta_x_holder = tf.placeholder(tf.float32, ())
        beta_y_l_holder = tf.placeholder(tf.float32, ())
        beta_y_f_holder = tf.placeholder(tf.float32, ())
        beta_y_c_holder = tf.placeholder(tf.float32, ())
        partial_loss_holder = tf.placeholder(tf.string, ())
        is_training = tf.placeholder(tf.bool, ())
        # model
        model = aan.AAN(images_holder, label_holder, low_bound_holder, up_bound_holder, epsilon_holder, is_training)
        model_loss_x, model_lx_dist, model_max_dist, _ = model.loss_x(beta_x_holder)
        model_loss_y, (model_ly_least, model_ly_fake, model_ly_clean), (model_ly_dist_least, model_ly_dist_fake, model_ly_dist_clean) =\
            model.loss_y(beta_y_l_holder, beta_y_f_holder, beta_y_c_holder)
        model_loss = model.loss(partial_loss_holder, model_loss_x, model_loss_y)
        model_optimization = model.optimization(model_loss)
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["low_bound_holder"] = low_bound_holder
        graph_dict["up_bound_holder"] = up_bound_holder
        graph_dict["epsilon_holder"] = epsilon_holder
        graph_dict["beta_x_holder"] = beta_x_holder
        graph_dict["beta_y_l_holder"] = beta_y_l_holder
        graph_dict["beta_y_f_holder"] = beta_y_f_holder
        graph_dict["beta_y_c_holder"] = beta_y_c_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # Load target classifier
        model._target.tf_load(sess, FLAGS.RESNET18_PATH, 'model.ckpt-5865')
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
        estop_count = 0
        
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
                    beta_x_holder: FLAGS.BETA_X,
                    beta_y_l_holder: FLAGS.BETA_Y_LEAST,
                    beta_y_f_holder: FLAGS.BETA_Y_FAKE,
                    beta_y_c_holder: FLAGS.BETA_Y_CLEAN,
                    is_training: True,
                    partial_loss_holder: FLAGS.PARTIAL_LOSS
                }
                # optimization
                fetches = [model_optimization, model_loss, model_loss_x, model_lx_dist, model_max_dist,
                        model_loss_y, model_ly_least, model_ly_fake, model_ly_clean, 
                        model_ly_dist_least, model_ly_dist_fake, model_ly_dist_clean,
                        merged_summary, model._target_fake_prediction]
                _, loss, l_x, Lx_dist, max_dist, l_y, Ly_least, Ly_fake, Ly_clean, Ly_dist_least, Ly_dist_fake, Ly_dist_clean, \
                    summary, fake_prediction = sess.run(fetches=fetches, feed_dict=feed_dict)
                
                #import pdb; pdb.set_trace()
                train_writer.add_summary(summary, train_idx)
                # Print info
                if train_idx % FLAGS.EVAL_FREQUENCY == (FLAGS.EVAL_FREQUENCY - 1):
                    print("Hyper-params info:")
                    print("Using Partial Loss:", FLAGS.PARTIAL_LOSS)
                    print("Pixel bound: [{:.4f}, {:.4f}]  Epsilon: {:.4f}".format(
                        -1.0*FLAGS.PIXEL_BOUND, FLAGS.PIXEL_BOUND, FLAGS.EPSILON))
                    print("Beta_X: {:.4f}  Beta_Y_LEAST: {:.4f}  Beta_Y_FAKE: {:.4f}  Beta_Y_CLEAN: {:.4f}".format(
                        FLAGS.BETA_X, FLAGS.BETA_Y_LEAST, FLAGS.BETA_Y_FAKE, FLAGS.BETA_Y_CLEAN
                    ))
                    print("Result:")
                    print("loss = {:.4f}  Loss x = {:.4f}  Loss y = {:.4f}".format(loss, l_x, l_y))
                    print("Loss y for least = {:.4f} Loss y for fake = {:.4f} Loss y for clean = {:.4f}".format(Ly_least, Ly_fake, Ly_clean))
                    print("Lx distance = {:.4f} Max pixel distance = {:.4f}".format(Lx_dist, max_dist))
                    print("Ly distance for least = {:.4f} Ly distance for fake = {:.4f} Ly distance for clean = {:.4f}".format(
                        Ly_dist_least, Ly_dist_fake, Ly_dist_clean))
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
            # Update bound
            if FLAGS.PIXEL_BOUND >= FLAGS.MIN_BOUND and FLAGS.PIXEL_BOUND <= FLAGS.MAX_BOUND and (epoch+1) % FLAGS.BOUND_CHANGE_EPOCHS == 0:
                FLAGS.PIXEL_BOUND =  FLAGS.PIXEL_BOUND * FLAGS.BOUND_CHANGE_RATE
            # Update epsilon
            if FLAGS.EPSILON >= FLAGS.MIN_EPSILON and FLAGS.EPSILON <= FLAGS.MAX_EPSILON and (epoch+1) % FLAGS.EPSILON_CHANGE_EPOCHS == 0:
                FLAGS.EPSILON =  FLAGS.EPSILON * FLAGS.EPSILON_CHANGE_RATE
            # Update Beta_x
            if FLAGS.BETA_X >= FLAGS.MIN_BETA_X and FLAGS.BETA_X <= FLAGS.MAX_BETA_X and (epoch+1) % FLAGS.BETA_X_CHANGE_EPOCHS == 0:
                FLAGS.BETA_X =  FLAGS.BETA_X * FLAGS.BETA_X_CHANGE_RATE
            # Update Beta_Y_FAKE
            if FLAGS.BETA_Y_FAKE >= FLAGS.MIN_BETA_Y_FAKE and FLAGS.BETA_Y_FAKE <= FLAGS.MAX_BETA_Y_FAKE and (epoch+1) % FLAGS.BETA_Y_FAKE_CHANGE_EPOCHS == 0:
                FLAGS.BETA_Y_FAKE =  FLAGS.BETA_Y_FAKE * FLAGS.BETA_Y_FAKE_CHANGE_RATE
            
                
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
            # validation
            print("\n**********************")
            print("Validation")
            valid_dict = test_info(sess, model, valid_writer, graph_dict, total_batch=total_valid_batch, valid=True)
            # early stopping
            if valid_dict["adv_acc"] < min_adv_acc:
                min_adv_acc = valid_dict["adv_acc"]
                estop_count = 0
            else:
                estop_count += 1
            if estop_count >= FLAGS.EARLY_STOPPING_THRESHOLD:
                print("Early Stopped.")
                break
                
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
