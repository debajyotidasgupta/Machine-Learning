# import modules
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def model(X_train, Y_train, hidden_layers=[], activation='logistic', lr=0.0001):
    
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers, 
        activation=activation,
        verbose=False, 
        learning_rate_init=lr
    )
    clf = clf.fit(X_train, Y_train)
    return clf


def tune_learning_rate(X_train, Y_train, X_test, Y_test):
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    hidden_layers = [[], [2], [6], [2,3], [3,2]]
    idx = 1
    scores = {}
    for hl in hidden_layers:
        scores[idx] = {}
        for learning_rate in lr:
            clf = model(X_train, Y_train, hidden_layers=hl, lr=learning_rate)
            score = clf.score(X_test, Y_test)
            scores[idx][learning_rate] = score
        idx += 1
    return scores

def best_model(X_train, Y_train, X_test, Y_test):
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    hidden_layers = [[], [2], [6], [2,3], [3,2]]
    scores = {}
    for learning_rate in lr:
        idx = 1
        scores[learning_rate] = {}
        for hl in hidden_layers:
            clf = model(X_train, Y_train, hidden_layers=hl, lr=learning_rate)
            score = clf.score(X_test, Y_test)
            scores[learning_rate][idx] = score
            idx += 1
    return scores
