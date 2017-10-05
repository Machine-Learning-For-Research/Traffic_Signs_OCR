import os
import numpy as np
import tensorflow as tf


def load_train_index(path):
    """
    加载训练数据索引
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
    labels = package[:, 1]
    return images, labels


def read_train_data(images, labels, n_class=43, im_width=48, im_height=48, batch_size=256, thread_num=64, capacity=1000):
    """
    将图片数据导入管道
    :param images: 图片路径集合
    :param labels: 图片标签集合
    :param im_width: 图片宽度
    :param im_height: 图片高度
    :param batch_size: 批大小
    :param thread_num: 线程数
    :param capacity: 缓冲容量
    :return: 图片和标签批次
    """
    queue = tf.train.slice_input_producer([images, labels])
    image_file = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, im_height, im_width)
    image = tf.image.per_image_standardization(image)
    label = queue[1]
    batch_image, batch_label = tf.train.batch([image, label], batch_size, thread_num, capacity)
    batch_label = tf.one_hot(batch_label, n_class)
    return batch_image, batch_label


if __name__ == '__main__':
    PATH = 'E:/ML/Lin/traffic signs/Training'
    images, labels = load_train_index(PATH)
    print('%d images, %d labels' % (len(images), len(labels)))

    batch_image, batch_label = read_train_data(images, labels)
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