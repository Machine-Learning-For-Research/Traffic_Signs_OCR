import tensorflow as tf

N_CLASS = 43


def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variables(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


def conv2d(x, W, b, strides, padding, activation=tf.nn.relu):
    x = tf.nn.conv2d(x, W, strides, padding) + b
    if activation is not None:
        x = activation(x)
    return x


def max_pool(x, ksize, strides, padding):
    return tf.nn.max_pool(x, ksize, strides, padding)


def fc(x, W, b, activation=tf.nn.relu):
    x = tf.matmul(x, W) + b
    if activation is not None:
        x = activation(x)
    return x

def inference(images, training):
    """
    模型推断
    :param images:
    :return:
    """
    x = images
    tf.summary.image('image', images)
    with tf.name_scope('conv_1'):
        W_conv1 = weight_variables([3, 3, 3, 32])
        b_conv1 = bias_variables([32])
        x = conv2d(x, W_conv1, b_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_2'):
        W_conv2 = weight_variables([3, 3, 32, 64])
        b_conv2 = bias_variables([64])
        x = conv2d(x, W_conv2, b_conv2, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_3'):
        W_conv3 = weight_variables([3, 3, 64, 128])
        b_conv3 = bias_variables([128])
        x = conv2d(x, W_conv3, b_conv3, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_4'):
        W_conv4 = weight_variables([3, 3, 128, 128])
        b_conv4 = bias_variables([128])
        x = conv2d(x, W_conv4, b_conv4, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    batch_size = int(images.get_shape()[0])
    x = tf.reshape(x, [batch_size, -1])

    with tf.name_scope('fc1'):
        W_fc1 = weight_variables([int(x.get_shape()[-1]), 2048])
        b_fc1 = bias_variables([2048])
        x = fc(x, W_fc1, b_fc1)
        x = tf.layers.batch_normalization(x, training=training)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variables([2048, 2048])
        b_fc2 = bias_variables([2048])
        x = fc(x, W_fc2, b_fc2)
        x = tf.layers.batch_normalization(x, training=training)

    with tf.name_scope('fc3'):
        W_fc3 = weight_variables([2048, N_CLASS])
        b_fc3 = bias_variables([N_CLASS])
        # noinspection PyTypeChecker
        x = fc(x, W_fc3, b_fc3, activation=None)

    return x


def calculate_loss(logits, labels):
    cross_entry = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entry)
    tf.summary.scalar('loss', loss)
    return loss


def get_train_step(loss, learning_rate=1e-3):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss)


def calculate_accuracy(logits, labels):
    """
    计算精度
    :param logits:
    :param labels:
    :return:
    """
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def calculate_accuracy_width_images(images, labels, training):
    logits = inference(images, training)
    return calculate_accuracy(logits, labels)
