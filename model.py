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
    with tf.name_scope('conv_1'):
        W = weight_variables([2, 2, 3, 32])
        b = bias_variables([32])
        x = conv2d(x, W, b, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_2'):
        W = weight_variables([2, 2, 32, 64])
        b = bias_variables([64])
        x = conv2d(x, W, b, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_3'):
        W = weight_variables([2, 2, 64, 128])
        b = bias_variables([128])
        x = conv2d(x, W, b, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv_4'):
        W = weight_variables([2, 2, 128, 128])
        b = bias_variables([128])
        x = conv2d(x, W, b, strides=[1, 1, 1, 1], padding='SAME')
        x = max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    batch_size = int(images.get_shape()[0])
    x = tf.reshape(x, [batch_size, -1])

    with tf.name_scope('fc1'):
        W = weight_variables([int(x.get_shape()[-1]), 2048])
        b = bias_variables([2048])
        x = fc(x, W, b)
        x = tf.layers.batch_normalization(x, training=training)

    with tf.name_scope('fc2'):
        W = weight_variables([2048, 2048])
        b = bias_variables([2048])
        x = fc(x, W, b)
        x = tf.layers.batch_normalization(x, training=training)

    with tf.name_scope('fc3'):
        W = weight_variables([2048, N_CLASS])
        b = bias_variables([N_CLASS])
        # noinspection PyTypeChecker
        x = fc(x, W, b, activation=tf.nn.softmax)

    return x


def calculate_loss(logits, labels):
    cross_entry = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = -tf.reduce_mean(cross_entry)
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
