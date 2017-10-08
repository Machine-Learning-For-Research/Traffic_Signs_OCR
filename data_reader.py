import os
import csv
import config
import numpy as np
import tensorflow as tf
import pandas as pd


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

def load_csv_index(csv_path):
    if not os.path.exists(csv_path):
        raise RuntimeError('File "%s" not found.')

    images, labels = [], []
    with open(csv_path) as f:
        for row, line in enumerate(csv.reader(f)):
            if row != 0:
                strs = line[0].split(';')
                # 这里添加相对路径
                images.append(os.path.join(config.load_test_path(), strs[0].strip()))
                labels.append(int(strs[-1].strip()))

    package = np.vstack([images, labels]).transpose()
    np.random.shuffle(package)
    images = package[:, 0]
    labels = package[:, 1].astype(int)
    return images, labels

def split_train_validate_data(images, labels, validate_rate=0.3):
    """
    分割训练集和验证集
    :param images:
    chaju:param labels:
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
    Train_PATH = config.load_train_path()
    csv_PATH = config.load_csv_path()

    train_images, train_labels = load_data_index(Train_PATH)
    test_images, test_labels = load_csv_index(csv_PATH)
    print('%d images, %d labels' % (len(train_images), len(train_labels)))
    print('%d images, %d labels' % (len(test_images), len(test_labels)))

    batch_train_image, batch_train_label = read_data(train_images, train_labels)
    batch_test_image, batch_test_label = read_data(test_images, test_labels)
    with tf.Session() as sess:
         coord = tf.train.Coordinator()
         threads = tf.train.start_queue_runners(sess, coord)
         try:
             value_image, value_label = sess.run([batch_test_image, batch_test_label])
             print('%d images, %d labels' % (len(value_image), len(value_label)))
         except tf.errors.OutOfRangeError as e:
             print('Error: %s' % str(e))
         finally:
             coord.request_stop()
         coord.join(threads)
