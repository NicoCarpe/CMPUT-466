#!/usr/bin/env python3.8

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold , learning_curve, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.inspection import DecisionBoundaryDisplay

X, y = make_moons(noise=0.3, random_state=1)

# use train_test_split twice to get out train, validation, and test datasets
# random shape shuffles data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


names = [
        "Logistic", #"Logistic Regression"
        "PSVM", #"Polynomial Support Vector Machine"
        "NN",   #"Neural Network"
]

classifiers = [
        LogisticRegression(
                penalty = "l2",
                max_iter = 1000
        ),
        SVC(
                kernel = "poly",
                max_iter = 1000
        ),
        MLPClassifier(
                max_iter = 1000,
                solver = "lbfgs"
        )
]

params = [
        # Logistic params  
        {'C'                   : [0.001, 0.01, 0.1],
        },

        # SVC params
        {'C'                   : [0.001, 0.01, 0.1], 
         'gamma'               : [0.001, 0.1, 1, 10]
        },

        # NN params
        {'alpha'               : [0.001, 0.01, 0.1],
         'hidden_layer_sizes'  : [(5, 2), (10,)],
         'activation'          : ['tanh', 'relu']
        }
]

# plot each result side by side
figure, axarr = plt.subplots(1, 3, figsize=(16, 8))
cm = plt.cm.Purples
accs = []
best_params = []
for i in range(len(classifiers)):
    # use grid search cross validation to tune hyperparameters with the specified parameters and classifier
    # the refit parameter specifies that the best estimator is set after training/validation is complete
    grid = GridSearchCV(
            estimator               = classifiers[i],
            param_grid              = params[i],
            scoring                 = "accuracy",
            refit                   = True
            )

    # train model and use cross validation to find the best parameters
    grid.fit(X_train, y_train)
    best_params.append(grid.best_params_)

    # get the accuracy of the model using the best parameters on the test dataset
    acc = grid.score(X_test, y_test)
    accs.append(acc)

    
    DecisionBoundaryDisplay.from_estimator(
            estimator               = grid, 
            X                       = X, 
            cmap                    = cm,
            alpha                   = 0.8, 
            ax                      = axarr[i],
            eps                     = 0.5
        )
    
    axarr[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    axarr[i].set_title(names[i])
    

### --- Baseline --- ###

baseline = DummyClassifier(strategy = "most_frequent")
baseline.fit(X_train, y_train)

# returns mean accuracy on the data X and labels y
base_acc = baseline.score(X_test, y_test)
print(f"Accuracy for Baseline: {base_acc}")

### --- Classifiers --- ###
for i in range(len(names)):
    print(f"Accuracy for {names[i]} with params {best_params[i]}: {accs[i]}")


plt.tight_layout
plt.savefig('Classification_Model_Accuracies.jpg')
    
    




