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