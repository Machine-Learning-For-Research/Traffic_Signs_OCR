# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:53:05 2017

@author: Linstancy
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd



def get_train_batch(train_path, img_width = 48, img_height = 48, batch_size = 50, num_threads = 64, capacity = 1000, standardization = True):
    
    if not os.path.exists(train_path):
        raise IOError("Files not found.")
        
    images = []
    labels = []
    
    #directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for d in range(43):
        label_dir = os.path.join(train_path, '/', 'd')
        #file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")]
        
        for f in label_dir:
            #if f.endswith(".png"):
            images.append(os.path.join(label_dir, '/', f))
            labels.append(int(d))
    #print('There are %d subdirectories' %(len(d)))
    print('There are %d training samples' %(len(images)))
    print('There are %d training class' %(len(labels)))

            
    package = np.array([images, labels])
    package = package.transpose()
    np.random.shuffle(package)
    images = package[:, 0]
    labels = package[:, 1]
    labels = [int(i) for i in labels]
    
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)
    queue = tf.train.slice_input_producer([images, labels])
    image_data = tf.read_file(queue[0])
    image = tf.image.decode_png(image_data, channels=3)
    
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    '''if standardization:
        image = tf.image.per_image_standeadization(image)
        '''
    label = queue[1]
    image_batch, label_batch = tf.train.batch([image, label], batch_size, num_threads, capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.cast(label_batch, tf.int32)
    
    
    
    return image_batch, label_batch

def get_eval_batch(test_path, img_width=48, img_height=48, batch_size=50, num_threads=64, capacity=1000, standardization=True):
    
    if not os.path.exists(test_path):
        raise IOError("Files not found.")
        
    images = []
    labels = []
    
    test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        images.append(os.path.join(test_path, file_name))
        labels.append(int(class_id))
        
    print('There are %d test samples' %(len(images)))
    print('There are %d training class' %(len(labels)))   
    
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)
    queue = tf.train.slice_input_producer([images, labels])
    image_data = tf.read_file(queue[0])
    image = tf.image.decode_png(image_data, channels=3)
    
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    '''if standardization:
        image = tf.image.per_image_standeadization(image)
    '''
    label = queue[1]
    image_batch, label_batch = tf.train.batch([image, label], batch_size, num_threads, capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.cast(label_batch, tf.int32)
    
    
    
    return image_batch, label_batch
            









































        
    
    