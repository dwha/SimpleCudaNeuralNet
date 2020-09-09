﻿# SimpleCudaNeuralNet
This is for studying both neural network and CUDA.

I focused on simplicity and conciseness while coding. That means there is no error handling but assertations. It is a self-study result for better understanding of back-propagation algorithm. It'd be good if this C++ code fragment helps someone who has an interest in deep learning. [CS231n](http://cs231n.stanford.edu/2017/syllabus) from Stanford provides a good starting point to learn deep learning.

## Status
#### Weight layers (w/ He initialization)
* 2D Convolutional
* Fully connected

#### Non-linearity
* Relu

#### Regularisation
* Max pooling
* Dropout
	
#### Loss
* Mean squared error
* Cross entropy loss

#### Optimizer 
* Adam

## Result
### Handwritten digit recognition
![ffCudaNn](https://user-images.githubusercontent.com/670560/91796552-735ee780-ec5b-11ea-88fc-0f0a343ce8d6.png)

After basic components for deep learning implemented, I built a handwritten digit recognizer using [MNIST database](http://yann.lecun.com/exdb/mnist/). A simple 2-layer FCNN(1000 hidden unit) could achieve 1.56% Top-1 error rate after 14 epochs which take less than 20 seconds of training time on RTX 2070 graphics card. (See [mnist.cpp](mnist.cpp))

### CIFAR-10 photo classification
![Screenshot from 2020-09-08 19-53-50](https://user-images.githubusercontent.com/670560/92467722-0e803000-f20d-11ea-9d20-7c4b828f4ff4.png)

In [cifar10.cpp](cifar10.cpp), you can find a VGG-like convolutional network which has 9 weight layers. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset without augmentation is used to train the model. It achieves 26.5% top-1 error rate after 50 epoches. It took 47 seconds per epoch on my RTX 2070.  If you try a larger model and have enough time to train you can improve it. Data augmentation and batch normalization will be helpful.

### Notes
- Even naive CUDA implementation easily speeds up by 700x more than single-core/no-SIMD CPU version.
- Double precision floating point on the CUDA kernels was 3~4x slower than single precision operations.
- Training performance is not comparable to PyTorch. PyTorch is much faster (x10~) to train the same model.
- Coding this kind of numerical algorithms is tricky and even hard to figure out if there is a bug or not. Thorough unit testing of every functions strongly recommended if you try.

