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
    model_loss_x, model_lx_dist, model_max_dist, _ = model.loss_x
    model_loss_y, model_ly_dist = model.loss_y
    model_loss = model.loss(model_loss_x, model_loss_y)
    fetches = [model._target_accuracy, model._target_adv_accuracy, model._target_fake_accuracy, model_loss, model_loss_x,
               model_lx_dist, model_max_dist, model_loss_y, model_ly_dist, graph_dict["merged_summary"]]
    if total_batch is None:
        if valid:
            total_batch = int(data.valid_size/FLAGS.BATCH_SIZE)
        else:
            total_batch = int(data.test_size/FLAGS.BATCH_SIZE)
    else: total_batch = total_batch

    acc = 0; adv_acc = 0; fake_acc = 0; loss = 0; l_x = 0; Lx_dist = 0; max_dist = 0; l_y = 0; Ly_dist = 0
    for idx in range(total_batch):
        if valid:
            batch_xs, batch_ys = data.next_valid_batch(FLAGS.BATCH_SIZE)
        else:
            batch_xs, batch_ys = data.next_test_batch(FLAGS.BATCH_SIZE)

        feed_dict = {
            graph_dict["images_holder"]: batch_xs,
            graph_dict["label_holder"]: batch_ys,
            graph_dict["partial_loss_holder"]: FLAGS.PARTIAL_LOSS,
            graph_dict["is_training"]: False
        }
        
        batch_acc, batch_adv_acc, batch_fake_acc, batch_loss, batch_l_x, batch_Lx_dist, batch_max_dist, batch_l_y, batch_Ly_dist, summary = \
            sess.run(fetches=fetches, feed_dict=feed_dict)
        test_writer.add_summary(summary, idx)
        acc += batch_acc
        adv_acc += batch_adv_acc
        fake_acc += batch_fake_acc
        loss += batch_loss
        l_x += batch_l_x
        Lx_dist += batch_Lx_dist
        max_dist += batch_max_dist
        l_y += batch_l_y
        Ly_dist += batch_Ly_dist
    acc /= total_batch
    adv_acc /= total_batch
    fake_acc /= total_batch
    loss /= total_batch
    l_x /= total_batch
    Lx_dist /= total_batch
    max_dist /= total_batch
    l_y /= total_batch
    Ly_dist /= total_batch

    #adv_images = X+adv_noises
    print('Original accuracy: {0:0.5f}'.format(acc))
    print('Faked accuracy: {0:0.5f}'.format(fake_acc))
    print('Attacked accuracy: {0:0.5f}'.format(adv_acc))
    print("Loss = {:.4f} Loss_x = {:.4f} Loss_y = {:.4f} Lx distance = {:.4f}  Max pixel distance = {:.4f}".format(
              loss, l_x, l_y, Lx_dist, max_dist))
    res_dict = {"acc": acc, 
                "adv_acc": adv_acc, 
                "loss": loss, 
                "l_x": l_x,
                "Lx_dist": Lx_dist,
                "max_dist": max_dist,
                "l_y": l_y, 
                "Ly_dist": Ly_dist
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
        partial_loss_holder = tf.placeholder(tf.bool)
        is_training = tf.placeholder(tf.bool)
        use_rerank_holder = tf.placeholder(tf.bool)

        model = aan.AAN(images_holder, label_holder, partial_loss_holder, is_training)
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
        graph_dict["partial_loss_holder"] = partial_loss_holder
        graph_dict["is_training"] = is_training
        graph_dict["merged_summary"] = merged_summary
        

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        model.tf_load(sess)
        # tensorboard writer
        test_writer = model_utils.init_writer(FLAGS.TEST_LOG_PATH, g)
        print("\nTest")
        test_info(sess, model, test_writer, graph_dict, total_batch=2)
        test_writer.close() 
        
        batch_xs, batch_ys = data.next_test_batch(FLAGS.BATCH_SIZE)
        adv_images = sess.run(model.prediction, feed_dict={images_holder: batch_xs, is_training: False})
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
        use_rerank_holder = tf.placeholder(tf.bool)
        rerank_holder = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES])
        partial_loss_holder = tf.placeholder(tf.bool)
        is_training = tf.placeholder(tf.bool)
        # model
        model = aan.AAN(images_holder, label_holder, partial_loss_holder, is_training)
        model_loss_x, model_lx_dist, model_max_dist, _ = model.loss_x
        model_loss_y, model_ly_dist = model.loss_y
        model_loss = model.loss(model_loss_x, model_loss_y)
        model_optimization = model.optimization(model_loss)
        merged_summary = tf.summary.merge_all()

        graph_dict = {}
        graph_dict["images_holder"] = images_holder
        graph_dict["label_holder"] = label_holder
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
        else:
            total_train_batch = int(data.train_size/FLAGS.BATCH_SIZE)
        for epoch in range(FLAGS.NUM_EPOCHS):
            start_time = time.time()
            for train_idx in range(total_train_batch):
                batch_xs, batch_ys = data.next_train_batch(FLAGS.BATCH_SIZE)
                feed_dict = {
                    images_holder: batch_xs,
                    label_holder: batch_ys,
                    is_training: True,
                    partial_loss_holder: FLAGS.PARTIAL_LOSS
                }
                # optimization
                fetches = [model_optimization, model_loss, model_loss_x, model_lx_dist, model_max_dist,
                           model_loss_y, model_ly_dist, merged_summary]
                _, loss, l_x, Lx_dist, max_dist, l_y, Ly_dist, summary = sess.run(fetches=fetches, feed_dict=feed_dict)
                train_writer.add_summary(summary, train_idx)
                # Print info
                if train_idx % FLAGS.EVAL_FREQUENCY == (FLAGS.EVAL_FREQUENCY - 1):
                    print("Using Partial Loss:", FLAGS.PARTIAL_LOSS)
                    print('loss = {:.4f} loss_x = {:.4f} loss_y = {:.4f} Lx distance = {:.4f} Max pixel distance = {:.4f}'.format(
                          loss, l_x, l_y, Lx_dist, max_dist))
                    model.tf_save(sess) # save checkpoint
                
            end_time = time.time()
            print('Eopch {} completed with time {:.2f} s'.format(epoch+1, end_time-start_time))
            # validation
            print("\nValidation")
            valid_dict = test_info(sess, model, valid_writer, graph_dict, total_batch=2, valid=True)
            # Update global variable
            if valid_dict["Ly_dist"] <= FLAGS.PARTIAL_THRESHOLD:
                FLAGS.PARTIAL_LOSS = False
                
            print()
        print("Optimization Finished!")

        model.tf_save(sess)
        print("Trained params have been saved to '%s'" % FLAGS.AE_PATH)

        train_writer.close() 
        valid_writer.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
