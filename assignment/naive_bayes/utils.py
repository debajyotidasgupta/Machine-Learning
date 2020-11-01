
"""
This python file contains the utility functions used in the assignment
"""

# Authors: Debajyoti Dasgupta <debajyotidasgupta6@gmail.com>
#          Siba Smarak Panigrahi <sibasmarak.p@gmail.com>

# import necessary modules 
from csv import reader                        
from math import sqrt, exp, pi
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from random import seed, randrange, shuffle
from sklearn.preprocessing import LabelEncoder

def train_test_split(dataset, train_size=0.8):
    '''
    This function splits the data into two sets - train and test
    Parameters:
    -----------
    dataset: dataset to be split
    train_size: the ratio of training data to original data

    Returns:
    --------
    X_train: training set
    X_test: test set
    '''

    shuffle(dataset)
    X_train = dataset[:int(0.8*len(dataset))]
    X_test = dataset[int(0.8*len(dataset)):]
    return X_train, X_test

def cross_validation_split(dataset, n_folds):
    '''
    This function splits the input data and returns the indices of the validation set for each iteration of cross-validation
    Parameters:
    -----------
    dataset: input data
    n_folds: number of times cross-validation is to be done (for a 5-fold cross validation, n_folds = 5)

    Returns:
    --------
    dataset_split: a n_fold length list of list of indices
                   each entry of this list is a set of indices to be used as validation set in the n_fols cross validation 
    '''

    dataset_split = list()
    l = len(dataset) // n_folds
    for i in range(n_folds):
        dataset_split.append(dataset[i*l:(i+1)*l])
    return dataset_split

def accuracy_metric(actual, predicted):
    '''
    This function evaluates the accuracy of predicted values with respect the actual values
    Accuracy is defined as the number of correct predictions divided by total length of predictions
    Parameters:
    -----------
    actual: list of actual values
    predicted: list of predicted values

    Returns:
    --------
    result: accuracy in %
    '''

    correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
    result = correct / float(len(actual)) * 100.0
	return result

def mean(numbers):
    '''
    This function returns the mean of the input numbers
    Parameters:
    -----------
    numbers: list of numbers whose mean is to be found

    Returns:
    --------
    mu: mean of the input nunmbers list
    '''

    mu = sum(numbers)/float(len(numbers))
	return mu

def stdev(numbers):
    '''
    This function returns the standard deviation of the input numbers
    Parameters:
    -----------
    numbers: list of numbers whose mean is to be found

    Returns:
    --------
    sigma: standard deviation of the input nunmbers list
    '''

	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    sigma = sqrt(variance)
	return sigma

def summarize_dataset(dataset):
    '''
    This functions returns the mean, standard deviation, number of entries, threshold value for outlier for each column of the dataset (characteristic values)
    Parameters:
    -----------
    dataset: input data for which the above mentioned characteristic values are to be evaluated for each column

    Returns:
    --------
    summaries: list of list of characteristic values of columns of dataset
    '''

    summaries = [(mean(column), stdev(column), len(column), mean(column)+3*stdev(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

def calculate_probability(x, mean, stdev):
    '''
    This function returns the probability density of the data point on a Gaussian with given input mean ad standard deviation
    Parameters:
    -----------
    x: data point whose probability density is to be evaluated
    mean: mean of the Gaussian
    stdev: standard deviation of the Gaussian

    Returns:
    --------
    prob: probability density of the data point x on the Gaussian with mu = mean and sigma = stdev
    '''

	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent