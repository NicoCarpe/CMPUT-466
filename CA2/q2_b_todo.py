#!/usr/bin/env python3

from utils import plot_data, generate_data, sigmoid, softmax, one_hot
import numpy as np
from numpy import transpose as T
from numpy.linalg import inv



"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    N = X.shape[0]

    # initialization
    alpha = 0.1
    epochs = 1000
    batch_size = 10

    w = np.zeros((X.shape[1], 1))
    b = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        for batch in range(int(np.ceil(N/batch_size))):

            X_batch = X[batch*batch_size: (batch+1)*batch_size]
            t_batch = t[batch*batch_size: (batch+1)*batch_size]
            t_batch = np.matrix(np.reshape(t_batch, (len(t_batch),1)))
            
            y_batch = np.zeros((X_batch.shape[0], 1))
            z = np.dot(X_batch, w) + np.ones((X_batch.shape[0], 1)) * b
            for i in range(z.shape[0]):
                y_batch[i] = sigmoid(z[i])
            
            w_grad = (1/batch_size) * np.sum(np.dot(X_batch.T, (y_batch - t_batch)), axis = 1)
            b_grad = (1/batch_size) * np.sum(y_batch - t_batch)
            

            # update weights and biases with the average gradients over the batch
            w -= alpha * w_grad
            b -= alpha * b_grad

    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    t_hat = np.zeros((X.shape[0],1))

    for i in range(X.shape[0]):
        z = np.dot(X[i], w) + b
        y = sigmoid(z)
        if y >= 0.5:
            t_hat[i] = 1 

    return t_hat


def train_softmax_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    N = X.shape[0]

    # initialization
    alpha = 0.1
    epochs = 1000
    batch_size = 10

    w = np.zeros((X.shape[1], 1))
    b = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        for batch in range(int(np.ceil(N/batch_size))):

            X_batch = X[batch*batch_size: (batch+1)*batch_size]
            t_batch = t[batch*batch_size: (batch+1)*batch_size]
            t_batch = np.matrix(np.reshape(t_batch, (len(t_batch),1)))
            
            y_batch = np.zeros((X_batch.shape[0], 1))
            z = np.dot(X_batch, w) + np.ones((X_batch.shape[0], 1)) * b
            for i in range(z.shape[0]):
                y_batch[i] = softmax(z[i])
            
                # choose class with highest prob (MAP Inference)   
                y_batch[i] = np.argmax(y_batch[i], axis = 1)
            
            w_grad = (1/batch_size) * np.sum(np.dot(X_batch.T, (y_batch - t_batch)), axis = 1)
            b_grad = (1/batch_size) * np.sum(y_batch - t_batch)
            

            # update weights and biases with the average gradients over the batch
            w -= alpha * w_grad
            b -= alpha * b_grad

    return w, b


def predict_softmax_regression(X, w, b):
    """
    Generate predictions by your linear regression classifier.
    """
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    z = np.dot(X, w) + b
    # z: Nsample x K
    K = 2
    
    t_hat = softmax(z)
        
    # choose class with highest prob (MAP Inference)    
    t_hat_best = np.argmax(t_hat, axis = 1)

    return t_hat_best


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    t = np.matrix(np.reshape(t, (len(t),1)))
    t_hat = np.matrix(np.reshape(t_hat, (len(t_hat),1)))
    
    # number of guesses correct divided by the number of targets
    acc = np.sum(np.equal(t, t_hat))/len(t)
    return acc


def main():
    # Dataset A
    # Softmax regression classifier
    X, t = generate_data("A")
    w, b = train_softmax_regression(X, t)
    t_hat = predict_softmax_regression(X, w, b)
    print("Accuracy of Softmax regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_softmax.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Softmax regression classifier
    X, t = generate_data("B")
    w, b = train_softmax_regression(X, t)
    t_hat = predict_softmax_regression(X, w, b)
    print("Accuracy of Softmax regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_softmax.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
