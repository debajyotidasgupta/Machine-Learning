"""
This python file reads the data from the PercentageIncreaseCOVIDWorldwide.csv
dataset and then forms regression tree out of it using the ID3 algorithm and
Variance as an measure of impurity
"""

# Authors: Debajyoti Dasgupta <debajyotidasgupta6@gmail.com>
#          Siba Smarak Panigrahi <sibasmarak.p@gmail.com>

import time
import random
import datetime
import graphviz
import pandas as pd
from math import log
from math import sqrt
from graphviz import Digraph
import matplotlib.pyplot as plt

#===============#
#   READ DATA   #
#===============#

def read_data():
    '''
    Returns
    -------
    df : A pandas dataframe consisting of the data
        read from the 'PercentageIncreaseCOVIDWorldwide.csv
        csv file
    '''
    df = pd.read_csv('PercentageIncreaseCOVIDWorldwide.csv').drop(index=0)
    return df

def get_date(date):
    '''
    Converts the date from a string to a UNIX time-stamp format

    Parameters
    ----------
    date : A string representing the given date in the format
            MM/ DD/ YYYY

    Returns
    -------
    unix: Converted date from string to unix format that
            is accepted by time, i.e a continuous function
    '''
    unix = time.mktime(datetime.datetime.strptime(date, "%m/%d/%Y").timetuple())
    return unix

def build_data(df):
    '''
    Converts the data from the pandas data freame format to 
    array of ddictionary objects
    
    Parameters
    ----------
    df: A pandas dataframe that contains the values read from the 
        input csv file. The data frame should have four attributes
        ["Date", "Confirmed", "Recovered", "Deaths"] and the target
        attribute, that is "Incease rate", Shape of the df should 
        be (n , 5)

    Returns
    -------
    data: It is the list that consists the values from the dataframe 
            converted to dictionary objects. Each object in the list 
            will represent exactly one sample of data. Sape of the
            data will be => length=n, each object will have 4 keys
    '''
    
    data = []
    for i in range(1, len(df['Confirmed'])+1):
        data.append(
            {
                'Date':           get_date(df['Date'][i]),
                'Confirmed':      df['Confirmed'][i],
                'Recovered':      df['Recovered'][i],
                'Deaths':         df['Deaths'][i],
                'Increase rate':  df['Increase rate'][i]
            }
        )
    return data

def train_test_split(df):
    '''
    This function will first convert the pandas dataframe 
    into array of dictionary objects then splits the data into 
    X_train and X_test using an 80-20 split for training : test
    
    Parameters
    ----------
    df: A pandas dataframe that contains the values read from the 
        input csv file. The data frame should have four attributes
        ["Date", "Confirmed", "Recovered", "Deaths"] and the target
        attribute, that is "Incease rate", Shape of the df should 
        be (n , 5)

    Returns
    -------
    X_train: [List]Contais the 60% data points collected from the 
                randomly shuffled dataset which will be used 
                for training. length=0.6*n
    
    X_valid: [List]Contais the 20% data points collected from the 
                randomly shuffled dataset which will be used 
                for cross validation. length=0.2*n

    X_test: [List]Contais the 20% data points collected from the 
                randomly shuffled dataset which will be used 
                for testing. length=0.2*n

    '''
    
    data = build_data(df)
    random.shuffle(data)
    X_train, X_valid, X_test = data[:int(0.6*len(data))], data[int(0.6*len(data)):int(0.8*len(data))], data[int(0.8*len(data)):]
    return X_train, X_valid, X_test

class node:
    '''
    This is the main node class of the decision tree
    This class contains the skeleton of the structure 
    of each node in the decision tree. 
    '''
    def __init__(self, attr, val, mean, mse):
        '''
        This function is the constructor to initialize the 
        values in the node object.
        
        Parameters
        ----------
        attr: [String] the decision attribute that has been selected 
                for this node on the basis of which the 
                children will be decided
        
        val: [Float] the value of the selected attribute on the 
                basis which the splitting of the children 
                nodes will be decided for the current node
        
        mean: [Float] mean of the attributes ( selected in the current 
                level ) of the training data, this will help in 
                making predictions at this node if it is a leaf

        mse: [Float] mean squared error of the attributes ( selected in the 
                current level ) of the training data, this will help 
                in making decisions for pruning
        '''
        self.attr = attr
        self.split = val
        self.mse = mse
        self.mean = mean
        self.left = None
        self.right = None

    def remove_children(self):
        '''
        This function is a helper function for the pruning step
        the following function removes the children of the current node

        '''
        self.right = None
        self.left = None
        self.attr = 'Increase rate'

    def restore(self, attr, left, right):
        '''
        This function will restore the  children nodes of the current 
        node during the pruning process if you decide not to remove 
        the cchildren of the current node
        
        Parameters
        ----------
        attr: the attribute of the current node
        left: left child of the current node
        right: right child of the current node 
        '''
        self.attr = attr
        self.left = left
        self.right = right

    def count_node(self):
        '''
        This function is a helper funcction which is used to recursively 
        count the number of nodes in the tree rooted at the given node 

        Returns
        -------
        num_nodes: this is the number of nodes in the sub tree rooted 
                    at the current node 
        '''
        num_nodes = 1
        if self.left!=None: num_nodes+=self.left.count_node()
        if self.right!=None: num_nodes+=self.right.count_node()
        return num_nodes

    def prune(self, decision_tree_root, cur_error, X_valid):
        '''
        This function is the main pruning function. This function 
        will first recursively prune the children of the current node
        then will decide whether to prune the correct node or not 
        
        Parameters
        ----------

        Returns
        -------

        '''
        if self.left==None and self.right==None:
            return 10**18
        if decision_tree.left!=None: cur_error = min(cur_error, self.left.prune(decision_tree_root, cur_error, X_valid))
        if decision_tree.right!=None: cur_error = min(cur_error, self.right.prune(decision_tree_root, cur_error, X_valid))

        # logic
        temp_left = self.left
        temp_right = self.right 
        temp_attr = self.attr 
        self.remove_children()

        err,_ = predict(decision_tree_root, X_valid)

        if err > cur_error:
            self.restore(temp_attr, temp_left, temp_right)
        return min(err, cur_error)

df = read_data()