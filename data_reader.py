import os
import config
import numpy as np
import tensorflow as tf


def load_data_index(path):
    """
    加载数据索引
    :param path: 训练数据路径
    :return: 训练数据集
    """
    if not os.path.exists(path):
        raise RuntimeError('The path "%s" not found.' % path)
    images, labels = [], []
    for chdir in os.listdir(path):
        dir_path = os.path.join(path, chdir)
        if not os.path.isdir(dir_path):
            continue
        sign_class = int(chdir)
        for img in os.listdir(dir_path):
            images.append(os.path.join(dir_path, img))
            labels.append(sign_class)
    package = np.vstack([images, labels]).transpose()
    np.random.shuffle(package)
    images = package[:, 0]
    labels = package[:, 1].astype(int)
    return images, labels


def split_train_validate_data(images, labels, validate_rate=0.3):
    """
    分割训练集和验证集
    :param images:
    :param labels:
    :param validate_rate:
    :return:
    """
    validate_size = int(len(images) * validate_rate)
    train_images, train_labels = images[:-validate_size], labels[:-validate_size]
    validate_images, validate_labels = images[-validate_size:], labels[-validate_size:]
    return train_images, train_labels, validate_images, validate_labels


def read_data(images, labels, im_width=48, im_height=48, batch_size=256, thread_num=64,
              capacity=1000, n_class=0):
    """
    将图片数据导入管道
    :param images: 图片路径集合
    :param labels: 图片标签集合
    :param im_width: 图片宽度
    :param im_height: 图片高度
    :param batch_size: 批大小
    :param thread_num: 线程数
    :param capacity: 缓冲容量
    :param n_class: 分类数, 传入则进行one_hot处理, 否则不处理
    :return: 图片和标签批次
    """
    queue = tf.train.slice_input_producer([images, labels])
    image_file = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, im_height, im_width)
    image = tf.image.per_image_standardization(image)
    label = queue[1]
    batch_image, batch_label = tf.train.batch([image, label], batch_size, thread_num, capacity)
    if n_class != 0:
        batch_label = tf.one_hot(batch_label, n_class, axis=1)
    return batch_image, batch_label


if __name__ == '__main__':
    PATH = config.load_train_path()
    images, labels = load_data_index(PATH)
    print('%d images, %d labels' % (len(images), len(labels)))

    batch_image, batch_label = read_data(images, labels)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            value_image, value_label = sess.run([batch_image, batch_label])
            print('%d images, %d labels' % (len(value_image), len(value_label)))
        except tf.errors.OutOfRangeError as e:
            print('Error: %s' % str(e))
        finally:
            coord.request_stop()
        coord.join(threads)
