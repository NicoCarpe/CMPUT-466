#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt
from utils import softmax, one_hot


def readMNISTdata():
    with open('t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here

    z = X @ W  
    # z: Nsample x K

    y_hot = one_hot(t, N_class)
    
    y = softmax(z)
    
    #y indexed by t
    loss_temp = y[np.arange(len(t)), t]

    # we don't want to have log(0) so truncate the loss to -log(1e-16)
    loss_tmp = np.where(loss_temp < 1e-16, loss_temp , 1e-16)

    loss = -np.sum(np.log(loss_tmp))
        
    # choose class with highest prob (MAP Inference)    
    y_best = np.argmax(y, axis = 1)
    acc = np.sum(np.equal(t, y_best))/len(t)

    return y_hot, y, loss, acc


def train(X_train, t_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # TODO Your code here

    # initialization
    W = np.zeros((X_train.shape[1],  N_class))
    # w: (d+1)x K
    
    train_losses = []
    val_accs = []

    W_best = None
    acc_best = 10000
    epoch_best = 0

    for epoch in range(1, MaxEpoch + 1):
        print(f"Epoch {epoch}")
        
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            t_batch = t_train[b*batch_size: (b+1)*batch_size]
            
            y_hot_batch, y_batch, loss_batch, _ = predict(X_batch, W, t_batch)
            
            loss_this_epoch += loss_batch
            
            W_grad = (1/batch_size) * np.dot(X_batch.T, (y_batch - y_hot_batch))
            W -= alpha * W_grad

        # divide by number of batches per epoch to get training loss
        loss_this_epoch = loss_this_epoch/int(np.ceil(N_train/batch_size)) 
        train_losses.append(loss_this_epoch)

        _, _, _, acc = predict(X_val, W, t_val)
        val_accs.append(acc)

        # we want to maximize accuracy
        if acc > acc_best:
            acc_best = acc
            W_best = W
            epoch_best = len(val_accs) 

    return epoch_best, acc_best,  W_best, train_losses, val_accs


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
print(X_test.shape[0], X_test.shape[1])
epoch_best, acc_best,  W_best, train_losses, val_accs = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)



print(f"Epoch with best validation performance : epoch {epoch_best}")
print(f"Epoch {epoch_best} validation performance (risk)  : {acc_best}")
print(f"Epoch {epoch_best} test performance (risk)        : {acc_test}")

epochs = range(0,len(val_accs))

# Learning curve of training loss 
plt.figure()
plt.plot(epochs, train_losses, color="blue")
plt.title("Training Loss Learning Curve")
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.tight_layout()
plt.savefig('Training_Loss_A.jpg')

# Learning curve of validation risk
plt.figure()
plt.plot(epochs, val_accs, color="red")
plt.title("Validation Accuracy Learning Curve")
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout()
plt.savefig('Validation_Accuracy_A.jpg')
