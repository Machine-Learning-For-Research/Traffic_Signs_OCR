import data_reader as reader
import tensorflow as tf
import datetime
import model
import config

TEST_PATH = config.load_csv_path()
LEARNING_RATE = 1e-3
MODEL_DIR = 'model_data'
IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50
N_CLASS = 43


def load_data():
    images, labels = reader.load_csv_index(TEST_PATH)
    batch_image, batch_label = reader.read_data(images, labels, IMAGE_WIDTH, IMAGE_HEIGHT, len(images), n_class=N_CLASS)
    return batch_image, batch_label


if __name__ == '__main__':
    test_image, test_label = load_data()
    test_logits = model.inference(test_image, training=False)
    # test_loss = model.calculate_loss(test_logits, test_labels)
    test_accuracy = model.calculate_accuracy(test_logits, test_label)
    # test_step = model.get_train_step(test_loss, LEARNING_RATE)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('Load last model params successfully.')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    # start training
    try:
        print('Start testing...')
        test_acc = sess.run(test_accuracy)
        time = str(datetime.datetime.now())
        print('Time %s, Test_Accuracy %s' % (time, test_acc))

    except tf.errors.OutOfRangeError as e:
        print('Error %s' % e)
    finally:
        coord.request_stop()
    coord.join(threads)
