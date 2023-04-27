#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = np.dot(X, w)
    

    # index the matmul products of loss and risk to get the scalar result
    loss = 1/(2*X.shape[0]) * np.dot(np.transpose(y_hat - y), (y_hat - y))[0][0]
    risk = 1/(X.shape[0]) * np.dot(np.transpose(abs(y_hat - y)), np.ones(y.shape[0]))[0]

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.random.randn(X_train.shape[1], 1)
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            w = w - alpha*(1/batch_size) * np.matmul(np.transpose(X_batch), y_hat_batch - y_batch)

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        # 2. Perform validation on the validation set by the risk
        # 3. Keep track of the best validation epoch, risk, and the weights

        # divide by number of batches per epoch to get training loss
        loss_this_epoch = loss_this_epoch/int(np.ceil(N_train/batch_size)) 
        #print(loss_this_epoch)
        losses_train.append(loss_this_epoch)

        _, _, risk = predict(X_val, w, y_val)
        risks_val.append(risk)

        # we want to minimize risk
        if risk < risk_best:
            risk_best = risk
            w_best = w
            # since we append the risk for each epoch the length of this list will equal the epoch number
            epoch_best = len(risks_val) 

    # Return some variables as needed
    return w_best, risk_best, epoch_best, losses_train, risks_val


############################
# Main code starts here
############################

# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

learning_rates =  [0.0001, 0.001, 0.01, 0.1]

for alpha in learning_rates:
    #alpha = 0.00001       # learning rate  [0.00001, 0.0001, 0.001, 0.005]
    MaxIter = 100        # Maximum iteration
    batch_size = 10      # batch size
    decay = 0.0          # weight decay


    # TODO: Your code here
    w_best, risk_val_best, epoch_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)

    # Perform test by the weights yielding the best validation performance
    y_hat_test, loss_test, risk_test = predict(X_test, w_best, y_test)


    # Report numbers and draw plots as required.

    print(f"Epoch with best validation performance : epoch {epoch_best}")
    print(f"Epoch {epoch_best} validation performance (risk)  : {risk_val_best}")
    print(f"Epoch {epoch_best} test performance (risk)        : {risk_test}")

    epochs = range(0,len(risks_val))

    # Learning curve of training loss 
    plt.figure()
    plt.plot(epochs, losses_train, color="blue")
    plt.title("Training Loss Learning Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.tight_layout()
    plt.savefig('Training_Loss_C_' + str(alpha) + '.jpg')

    # Learning curve of validation risk
    plt.figure()
    plt.plot(epochs, risks_val, color="red")
    plt.title("Validation Risk Learning Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Risk')
    plt.tight_layout()
    plt.savefig('Validation_Risk_C_' + str(alpha) + '.jpg')