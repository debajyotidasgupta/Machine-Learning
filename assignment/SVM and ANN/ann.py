# import modules
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def model(X_train, Y_train, hidden_layers=[], activation='logistic', lr=0.0001):
    '''
    This function creates the MLP classifier and
    fits the model on  the  input  training data

    Parameters
    ----------
    X_train: Training dataset
    Y_train: Target of the training dataset
    hidden_layers: tuples containing
    '''
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers, 
        solver='sgd',
        activation=activation,
        verbose=False, 
        learning_rate_init=lr
    )
    clf = clf.fit(X_train, Y_train)
    return clf


def tune_learning_rate(X_train, Y_train, X_test, Y_test, best_model, best_score, activation):
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    hidden_layers = [[], [2], [6], [2,3], [3,2]]
    idx = 1
    scores = {}
    for hl in hidden_layers:
        scores[idx] = {}
        for learning_rate in lr:
            clf = model(X_train, Y_train, hidden_layers=hl, lr=learning_rate, activation=activation)
            score = clf.score(X_test, Y_test)
            if score > best_score:
                best_model = clf
                best_score = score
            scores[idx][learning_rate] = score
        idx += 1
    return scores, best_model, best_score

def tune_model(X_train, Y_train, X_test, Y_test, best_model, best_score, activation):
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    hidden_layers = [[], [2], [6], [2,3], [3,2]]
    scores = {}
    for learning_rate in lr:
        idx = 1
        scores[learning_rate] = {}
        for hl in hidden_layers:
            clf = model(X_train, Y_train, hidden_layers=hl, lr=learning_rate, activation=activation)
            score = clf.score(X_test, Y_test)
            if score > best_score:
                best_model = clf
                best_score = score
            scores[learning_rate][idx] = score
            idx += 1
    return scores, best_model, best_score
