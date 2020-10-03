"""
This python file contains the utility functions that will help in the 
construction of the decision tree and printing the tree. Here variance 
is used as an measure of impurity
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
