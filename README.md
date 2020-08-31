﻿# SimpleCudaNeuralNet
This is for studying both neural network and CUDA.

I focused on simplicity and conciseness while coding. It is a self-studying result for better understanding of back-propagation algorithm. It'd be good if this C++ code fragment helps someone who have an interest in deep learning.

## Status
#### Layer
* Fully connected
	
#### Non-linearity
* Relu

#### Loss
* Sum of Squares 
* Softmax

#### Optimizer 
* Adam

## Result
Even naive CUDA implementation easily speeds up by more 200x than single core CPU version.

It is easy to build handwritten digit recognizer using [MNIST database](http://yann.lecun.com/exdb/mnist/). My first attempt on 2-layer FCNN (1000 hidden unit) could achieve 1.56% Top-1 error rate after 14 epochs which take less than 30 seconds of training time.