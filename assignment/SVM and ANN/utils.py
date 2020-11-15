from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# pick 80% data randomly as training set and rest as test set
def split_data(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[: , :41], data[41], train_size=0.8,)
    print('Size of X_train = {}\nSize of X_test = {}\nSize of Y_train = {}\nSize of Y_test = {}'.format(
        X_train.shape, 
        X_test.shape, 
        Y_train.shape, 
        Y_test.shape
    ))
    return X_train, X_test, Y_train, Y_test


def print_ker_acc(train_acc_C, test_acc_C, kernel_name_switcher):
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

def best_ker_acc(train_acc_C, test_acc_C, kernel_name_switcher):
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


def plot_scores(scores, mapper):
    for key in list(scores.keys()):
        lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        plt.plot(lr, list(scores[key].values()))
        plt.title(mapper[str(key)])
        plt.xlabel('learning rate')
        plt.xscale('log')
        plt.ylabel('accuracy')

def plot_scores_3(scores, mapper_3):
    for key in list(scores.keys()):
        xaxis = [mapper_3[str(i)] for i in list(scores[key].keys())]
        yaxis = list(scores[key].values())
        plt.title('learning rate = {:0.5f}'.format(key))
        plt.plot(np.linspace(1,5,5), yaxis)
        plt.xticks(np.linspace(1,5,5), xaxis, rotation=45)
        plt.xlabel(' (Hidden Layers, units) | units in a tuple')
        plt.ylabel('accuracy')