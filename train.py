import data_reader as reader
import tensorflow as tf
import datetime
import model
import util
import os

TRAIN_PATH = util.load_train_path()
LOG_DIR = 'log'
MODEL_DIR = 'model_data'
N_CLASS = 43
IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50
BATCH_SIZE = 128
VALIDATE_RATE = 0.001
EPOCH = 10
LEARNING_RATE = 1e-3


def load_data():
    # load all data index
    images, labels = reader.load_data_index(TRAIN_PATH)
    # split train and validate data
    t_images, t_labels, v_images, v_labels = reader.split_train_validate_data(images, labels, VALIDATE_RATE)
    # transform train data to batch data
    t_image, t_label = reader.read_data(t_images, t_labels, IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, n_class=N_CLASS)
    # transform validate data to batch data
    v_image, v_label = reader.read_data(v_images, v_labels, IMAGE_WIDTH, IMAGE_HEIGHT, len(v_images), n_class=N_CLASS)
    return len(t_images), t_image, t_label, v_image, v_label


if __name__ == '__main__':
    # load train and validate data
    train_size, train_image, train_label, validate_image, validate_label = load_data()
    # ready something about training
    train_logits = model.inference(train_image, True)
    train_loss = model.calculate_loss(train_logits, train_label)
    train_accuracy = model.calculate_accuracy(train_logits, train_label)
    validate_accuracy = model.calculate_accuracy_width_images(validate_image, validate_label, False)
    train_step = model.get_train_step(train_loss, LEARNING_RATE)

    # initialize summary
    summary_merge = tf.summary.merge_all()
    graph = tf.get_default_graph()
    train_writer = tf.summary.FileWriter(LOG_DIR, graph)
    validate_writer = tf.summary.FileWriter(LOG_DIR, graph)

    # initialize session and saver
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('Load last model params successfully.')

    # initialize coord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    # start training
    try:
        print('Start training...')
        max_iterator = int(train_size * EPOCH / BATCH_SIZE)
        for step in range(1, max_iterator + 1):
            _, loss, accuracy, summary_str = sess.run([train_step, train_loss, train_accuracy, summary_merge])
            # _, loss, accuracy, summary_str, a, b = sess.run([train_step, train_loss, train_accuracy, summary_merge, train_image, train_label])
            if step % 10 == 0 or step == max_iterator:
                time = str(datetime.datetime.now())
                epoch = int(step * BATCH_SIZE / train_size)
                print('Time %s, Epoch %d, Step: %d, Accuracy %s, Loss %s' % (time, epoch, step, accuracy, loss))
            if step % 50 == 0 or step == max_iterator:
                train_writer.add_summary(summary_str, step)
                train_writer.flush()
                pass
            if step % 100 == 0 or step == max_iterator:
                saver.save(sess, os.path.join(MODEL_DIR, 'model'), step)
    except tf.errors.OutOfRangeError as e:
        print('Error %s' % e)
    finally:
        coord.request_stop()
    coord.join(threads)
