# import modules
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy

# there is no missing value column
# RB: Ready Biodegradable -> assigned as True (356)
# NRB: Not Ready Biodegradable -> assigned as False (699)
# shape: (1055, 42)
# last column is the target
data = pd.read_csv('biodeg.csv', sep=';', header=None)
data[41] = data[41] == 'RB'

# pick 80% data randomly as training set and rest as test set
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[: , :41], data[41], train_size=0.8,)
print('Size of X_train = {}\nSize of X_test = {}\nSize of Y_train = {}\nSize of Y_test = {}'.format(X_train.shape
                                                                                                    , X_test.shape, 
                                                                                                    Y_train.shape, 
                                                                                                    Y_test.shape))

kernel_name_switcher = {
    'rbf': 'Radial Basis \nFunction',
    'linear': 'Linear',
    'poly': 'Quadratic'
}

def svm_classifiers(X_train, Y_train, X_test, Y_test, kernels=['rbf']):
    '''
    This function takes a list of kernels, trains a SVM classifier and returns the accuracy lists of train and test accuracies
    Parameters:
    -----------
    X_train: Training set data
    Y_train: Training set targets
    X_test: Testing set data
    Y_test: Testing set targets
    kernels: a list of kernels

    Returns:
    --------
    train_acc: a dictionary with (kernel, training accuracy) as key-value pairs
    test_acc: a dictionary with (kernel, testing accuracy) as key-value pairs
    '''

    train_acc, test_acc = {}, {}
    for kernel in kernels:
        classifier = SVC(kernel=kernel, degree=2)
        classifer = classifier.fit(X_train, Y_train)
        if kernel == 'poly':
            kernel = 'quadratic'
        train_acc[kernel] = classifier.score(X_train, Y_train)
        test_acc[kernel] = classifier.score(X_test, Y_test)
    print('####################### Train Accuracies #######################')
    print(train_acc)
    print('\n')
    print('####################### Test Accuracies #######################')
    print(test_acc)
    return train_acc, test_acc

def find_best_C(X_train, Y_train, X_test, Y_test, kernels):
    '''
    This function returns the train and test accuracies for an input list of kernels 
    with C values of 1.e-03 to 1.e+03 
    Parameters:
    -----------
    X_train: Training set data
    Y_train: Training set targets
    X_test: Testing set data
    Y_test: Testing set targets
    kernels: a list of kernels

    Returns:
    --------
    train_acc: a dictionary with (C-value, training accuracy) as key-value pairs for each kernel
    test_acc: a dictionary with (C-value, testing accuracy) as key-value pairs for each kernel
    '''

    train_acc, test_acc = {}, {}
    for kernel in kernels:
        Clist = np.logspace(-3, 3, num=7)
        train_acc[kernel], test_acc[kernel] = {}, {}
        for C in Clist:
            classifier = SVC(kernel=kernel, degree=2, C=C)
            classifer = classifier.fit(X_train, Y_train)
            train_acc[kernel][C] = classifier.score(X_train, Y_train)
            test_acc[kernel][C] = classifier.score(X_test, Y_test)
    return train_acc, test_acc

def print_acc(train_acc_C, test_acc_C):
    '''
    This function prints the result of various kernels and 
    corresponding C-values present in given dictionaries in a tabular form
    Parameters:
    -----------
    train_acc_C: a dictionary with (C-value, training accuracy) as key-value pairs for each kernel
    test_acc_C: a dictionary with (C-value, testing accuracy) as key-value pairs for each kernel
    '''

    print('C \t     Train accuracy   Test accuracy')
    print('---------------------------------------------------')
    for ker in list(train_acc_C.keys()):
        print('\n' + kernel_name_switcher[ker] + ':\n')
        for c in list(train_acc_C[ker].keys()):
            print(str(c) + '\t\t' + '{:0.3f}'.format(train_acc_C[ker][c]) + '\t\t' + '{:0.3f}'.format(test_acc_C[ker][c]))
    print('---------------------------------------------------')

def best_acc(train_acc_C, test_acc_C):
    '''
    This function prints the best C value for each kernel from input train_acc and test_acc dictionaries
    Parameters:
    -----------
    train_acc_C: a dictionary with (C-value, training accuracy) as key-value pairs for each kernel
    test_acc_C: a dictionary with (C-value, testing accuracy) as key-value pairs for each kernel
    '''

    print('Best Accuracies:')
    print('----------------')
    for ker in list(train_acc_C.keys()):
        print('\n' + kernel_name_switcher[ker] + ':\n')
        maxc = max(test_acc_C[ker], key=test_acc_C[ker].get)
        print('Maximum Test Accuracy occurs with C = {}'.format(maxc))
        print('Corresonding Train Accuracy: {:0.3f}'.format(train_acc_C[ker][maxc]))
        print('Corresonding Test Accuracy: {:0.3f}'.format(test_acc_C[ker][maxc]))

train_acc, test_acc = svm_classifiers(X_train, Y_train, X_test, Y_test, ['rbf', 'linear', 'poly'])
train_acc_C, test_acc_C = find_best_C(X_train, Y_train, X_test, Y_test, ['rbf', 'linear', 'poly'])
print_acc(train_acc_C, test_acc_C)
best_acc(train_acc_C, test_acc_C)

