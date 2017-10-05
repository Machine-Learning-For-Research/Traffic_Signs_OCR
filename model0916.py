# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 14:41:31 2017

@author: Linstancy
"""

import tensorflow as tf 
import numpy as np
import cv2
import csv
import os
import skimage.data
import skimage.transform
from skimage import io
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import pandas as pd
import input_data

train_path= 'C:/Users/Linstancy/Desktop/Model Code/datasets/train'
test_path = 'C:/Users/Linstancy/Desktop/Model Code/datasets/test'
#train_data_dir = os.path.normpath("BelgiumTS/Training")
#test_data_dir = os.path.normpath("BelgiumTS/Testing")
batch_size = 50
rate = 0.001
EPOCHS=1000


'''
def load_train_data(data_dir, is_train):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    if is_train:
        
        directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        # Loop through the label directories and collect the data in
        # two lists, labels and images.
        y_train = []
        x_train = []
        
        for d in directories:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".ppm")]
            
            # For each label, load it's images and add them to the images list.
            # And add the label number (i.e. directory name) to the labels list.
            for f in file_names:
                x_train.append(skimage.data.imread(f))
                y_train.append(int(d))
                
        return x_train, y_train

def load_test_data(data_dir, is_test):
    
    if is_test:
        
        directories=[d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]
    
        y_test = []
        x_test = []
        
        for d in directories:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".ppm")]
            
            # For each label, load it's images and add them to the images list.
            # And add the label number (i.e. directory name) to the labels list.
            for f in file_names:
                x_test.append(skimage.data.imread(f))
                y_test.append(int(d))
        
        return y_test, x_test

def load_test_data(data_dir, is_test):
    
    test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')
    # Load test dataset
    x_test = []
    y_test = []

    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('GTSRB/Test/Images/', file_name)
        x_test.append(io.imread(img_path))
        y_test.append(class_id)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_test, y_test

def weights(shape):
    return tf.Variable(tf.truncated_normal(shape=[1,1,3,32]));

def conv2d(x, W, b, strides=[1,1,1,1], activation=tf.nn.relu):
    x = tf.nn.conv2d(x, W, strides, padding='SAME') + b
    if activation is not None:
        x = activation(x)
    return x
'''
def lsy_model(x, train):
    
    with tf.name_scope('conv_1'):
        w1 = tf.Variable(tf.truncated_normal(shape=[1, 1, 3, 32]))
        b1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32))
        conv_1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')+ b1
        relu1 = tf.nn.relu(conv_1)
        
    with tf.name_scope('conv_2'):
        w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64]))
        b2 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32))
        conv_2 = tf.nn.conv2d(relu1, w2, strides=[1, 1, 1, 1], padding='SAME')+ b2
        pool2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.relu(pool2)
    
    with tf.name_scope('conv_3'):
        w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 192]))
        b3 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
        conv_3 = tf.nn.conv2d(relu2, strides=[1, 1, 1, 1], w3, padding='SAME') + b3
        pool3 = tf.nn.max_pool(conv_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        relu3 = tf.nn.relu(pool3)
        
    with tf.name_scope('conv4'):
        w4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 192]))
        b4 = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
        conv_4 = tf.nn.conv2d(relu3 ,w4, strides=[1, 1, 1, 1], padding='SAME')+ b4
        pool4 = tf.nn.max_pool(conv_4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        relu4 = tf.nn.relu(pool4)
        
    fc0=flatten(relu4)
    
    with tf.name_scope('fc1'):
        w5=tf.Variable(tf.truncated_normal(shape=[6912, 2048]))
        b5 = tf.Variable(tf.constant(0.0, shape=[2048], dtype=tf.float32))
        fc1 = tf.matmul(fc0, w5) + b5
        fc1 = tf.nn.relu(fc1)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.name_scope('fc2'):
        w6 = tf.Variable(tf.truncated_normal(shape = [2048, 1024]))
        b6 = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32))
        fc2 = tf.matmul(fc1, w6) + b6
        fc2 = tf.nn.relu(fc2)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.name_scope('fc3'):
        w7 = tf.Variable(tf.truncated_normal(shape = [1024, 256]))
        b7 = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32))
        fc3 = tf.matmul(fc2, w7)+ b7
        fc3 = tf.nn.relu(fc3)
        if train: fc2 = tf.nn.dropout(fc3, 0.5)
        
    with tf.name_scope('fc4'):
        w8 = tf.Variable(tf.truncated_normal(shape = [256, 43]))
        b8 = tf.Variable(tf.constant(0.0, shape = [43], dtype = tf.float32))
        logit = tf.matmul(fc3, w8)+ b8
        
        if train: logit = tf.nn.dropout(logit, 0.5)
        
        
    return logit

def one_hot(data):
    size = data.shape[0]
    deep = np.max(data.astype(int)) + 1
    result = np.zeros([size, deep])
    result[np.arange(size), data] = 1
    return result

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [batch_size, 48, 48, 3], name = 'x-input')
    y = tf.placeholder(tf.int32, [batch_size, 43],name = 'y-input')
    
    
images_train_batch, labels_train_batch = input_data.get_train_batch(train_path, img_width = 48, img_height = 48, batch_size = 50, num_threads = 64, capacity = 1000, standardization = True)
images_train_batch = np.array(images_train_batch)
labels_train_batch = np.array(labels_train_batch)
#labels_train_batch = one_hot(labels_train_batch)
images_test_batch, labels_test_batch = input_data.get_eval_batch(test_path, img_width = 48, img_height = 48, batch_size = 50, num_threads = 64, capacity = 1000, standardization = True)
images_test_batch = np.array(images_test_batch)
labels_test_batch = np.array(labels_test_batch)
#labels_test_batch = one_hot(labels_test_batch)
logit = lsy_model(images_train_batch,train = True)

#global_step = tf.placeholder(tf.int32)#global_step:i*num_examples+offset

with tf.name_scope('optimizer'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = labels_train_batch)
    loss_operation = tf.reduce_mean(cross_entropy)
    learning_rate = tf.train.exponential_decay(rate, 1000, 1, 0.99, staircase = True, name = None)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(one_hot_y, 1), tf.argmax(logits, 1), num_classes)        

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    print("Training...")
    for i in range(EPOCHS):
        '''x_train_epoch, y_train_epoch = shuffle(x_train, y_train) 
        num_examples = len(x_train)
        #start_time = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x,batch_y= x_train_epoch[offset:end],y_train_epoch[offset:end]'''
        _, loss, train_acc = sess.run([training_operation, loss_operation, accuracy_operation])
            
        if i % 100 == 0:
            print(" %d training step(s),training accuracy is %g" % (i, train_acc))
        
    test_acc=sess.run(accuracy_operation,feed_dict={x:images_test_batch, y:labels_test_batch})
    print("After training step(s), test accuracy is %g" % (test_acc))
    
        
    
    
    
    
        
        
    
    
        
        
                           












































