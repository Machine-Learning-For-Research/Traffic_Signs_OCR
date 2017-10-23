#Traffic Sign Recognition

This is my implementation of Traffic Sign Recognition Project by convolutional neural networks on Tensorflow, an open source machine distributed framework. The model alternates 5 convolutional with max-pooling layers, followed by 3 full connected layers. The activation function is Rectified linear unit (Relu). Dropout is as a normalization method of avoiding overfitting. Finally, Softmax is used for producting class-specific probabilities.
Training process is conducted with stochastic gradient descay by 50 epochs on German Traffic Sign Recognition Benchmark (GTSRB). The dataset contain 39290 training images and 12690 test images. After training, we save the last model parameter ans restore to test the model performance on 12690 test samples, and or so 95% recognition accuracy is obtained. If possible, you can attempt to change train epoch or train method to observe the model performance. 

I hope this project could help people who is learning deep learning (just like me).
