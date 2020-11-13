# import modules
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

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

mapper = {
    '1': 'no hidden layers',
    '2': '1 hidden layer with 2 nodes',
    '3': '1 hidden layer with 6 nodes',
    '4': '2 hidden layers with 2 and 3 nodes respectively',
    '5': '2 hidden layers with 3 and 2 nodes respectively'
          }
mapper_3 = {
    '1': '0, ()',
    '2': '1, (2)',
    '3': '1, (6)',
    '4': '2, (2,3)',
    '5': '2, (3,2)'
}

print('Size of input: {}'.format(X_train.shape[1]))
print('Size of output: {}'.format(1))

def model(hidden_layers=[], activation='logistic', lr=0.0001):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, 
                        verbose=False, learning_rate_init=lr)
    clf = clf.fit(X_train, Y_train)
    return clf

print('All the following results with learning rate = 0.0001\n')

# with 0 hidden layer
clf = model()
print('With 0 hidden layer')
print('-------------------')
print('Accuracy:\t{:0.3f}\n'.format(clf.score(X_test, Y_test)))
del clf

# with 1 hidden layer with 2 nodes
clf = model(hidden_layers=[2])
print('With 1 hidden layer with 2 nodes')
print('--------------------------------')
print('Accuracy:\t{:0.3f}\n'.format(clf.score(X_test, Y_test)))
del clf

# with 1 hidden layer with 6 nodes
clf = model(hidden_layers=[6])
print('With 1 hidden layer with 6 nodes')
print('--------------------------------')
print('Accuracy:\t{:0.3f}\n'.format(clf.score(X_test, Y_test)))
del clf

# with 2 hidden layers with 2 and 3 nodes respectively
clf = model(hidden_layers=[2,3])
print('With 2 hidden layers with 2 and 3 nodes respectively')
print('----------------------------------------------------')
print('Accuracy:\t{:0.3f}\n'.format(clf.score(X_test, Y_test)))
del clf

# with 2 hidden layers with 3 and 2 nodes respectively
clf = model(hidden_layers=[3,2])
print('With 2 hidden layers with 3 and 2 nodes respectively')
print('----------------------------------------------------')
print('Accuracy:\t{:0.3f}\n'.format(clf.score(X_test, Y_test)))
del clf

def part2():
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    hidden_layers = [[], [2], [6], [2,3], [3,2]]
    idx = 1
    scores = {}
    for hl in hidden_layers:
        scores[idx] = {}
        for learning_rate in lr:
            clf = model(hidden_layers=hl, lr=learning_rate)
            score = clf.score(X_test, Y_test)
            scores[idx][learning_rate] = score
        idx += 1
    return scores

def part3():
    lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    hidden_layers = [[], [2], [6], [2,3], [3,2]]
    scores = {}
    for learning_rate in lr:
        idx = 1
        scores[learning_rate] = {}
        for hl in hidden_layers:
            clf = model(hidden_layers=hl, lr=learning_rate)
            score = clf.score(X_test, Y_test)
            scores[learning_rate][idx] = score
            idx += 1
    return scores

scores = part2()
scores_3 = part3()

def plot_scores(scores):
    for key in list(scores.keys()):
        lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        plt.plot(lr, list(scores[key].values()))
        plt.title(mapper[str(key)])
        plt.xlabel('learning rate')
        plt.xscale('log')
        plt.ylabel('accuracy')
        plt.show()

plot_scores(scores)

def plot_scores_3(scores):
    for key in list(scores.keys()):
        xaxis = [mapper_3[str(i)] for i in list(scores[key].keys())]
        yaxis = list(scores[key].values())
        plt.title('learning rate = {:0.5f}'.format(key))
        plt.plot(np.linspace(1,5,5), yaxis)
        plt.xticks(np.linspace(1,5,5), xaxis, rotation=45)
        plt.xlabel(' (Hidden Layers, units) | units in a tuple')
        plt.ylabel('accuracy')
        plt.show()

plot_scores_3(scores_3)

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

    print('\nC \t     Train accuracy   Test accuracy')
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

    print('\nBest Accuracies:')
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