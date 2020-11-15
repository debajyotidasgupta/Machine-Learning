import time
import warnings
import pandas as pd
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from utils import split_data, print_ker_acc, best_ker_acc
from svm import svm_classifiers, find_best_C
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # there is no missing value column
    # RB: Ready Biodegradable -> assigned as True (356)
    # NRB: Not Ready Biodegradable -> assigned as False (699)
    # shape: (1055, 42)
    # last column is the target
    start = time.time()
    
    print("\n ============= READING DATA ============ \n")
    data = pd.read_csv('biodeg.csv', sep=';', header=None)
    data[41] = data[41] == 'RB'
    scaler = StandardScaler()
    data[:41] = scaler.fit_transform(data[:41])
    print("Time elapsed  =  {} ms".format(time.time()-start))
    print("\n ============= DATA READ ============ \n\n")

    print("============= SPLITTING DATASET INTO TRAIN TEST ============\n")
    X_train, X_test, Y_train, Y_test = split_data(data)
    print("\nTime elapsed  =  {} ms\n".format(time.time()-start))
    print("============= TRAIN TEST SPLIT COMPLETE ============\n")

    print("===================== SOLVING Q1 ===================\n\n")
    print("\
        ######################################\n\
        #                                    #\n\
        #          SVM CLASSIFIER            #\n\
        #                                    #\n\
        ######################################\n\
        \n")
    kernel_name_switcher = {
        'rbf': 'Radial Basis Function',
        'linear': 'Linear',
        'poly': 'Quadratic'
    }

    print('\nkernel_name_switcher = {\n\
        \'rbf\': \'Radial Basis Function\',\n\
        \'linear\': \'Linear\',\n\
        \'poly\': \'Quadratic\'\n\
    }\n\n')

    print('============= IMPLEMENT BINARY SVM CLASSIFIER ===================\n')
    train_acc, test_acc = svm_classifiers(X_train, Y_train, X_test, Y_test, ['rbf', 'linear', 'poly'])
    print("\nTime elapsed  =  {} ms\n".format(time.time()-start))
    print('\n============= FINDING BEST C VALUER FOR SVM CLASSIFIER ==========\n')
    train_acc_C, test_acc_C = find_best_C(X_train, Y_train, X_test, Y_test, ['rbf', 'linear', 'poly'], 1)
    print_ker_acc(train_acc_C, test_acc_C, kernel_name_switcher)
    best_ker_acc(train_acc_C, test_acc_C, kernel_name_switcher)

    print("\nTime elapsed  =  {} ms\n".format(time.time()-start))
    print("\n============== SOLVED Q1 ==============\n")

    print("\n============== SOLVING Q2 ==============\n")
    print("\
        ######################################\n\
        #                                    #\n\
        #          ANN CLASSIFIER            #\n\
        #                                    #\n\
        ######################################\n\
        \n")

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

    print('mapper = {\n\
        \'1\': \'no hidden layers\',\n\
        \'2\': \'1 hidden layer with 2 nodes\',\n\
        \'3\': \'1 hidden layer with 6 nodes\',\n\
        \'4\': \'2 hidden layers with 2 and 3 nodes respectively\',\n\
        \'5\': \'2 hidden layers with 3 and 2 nodes respectively\'\n\
    }\n')

    print('Size of input: {}'.format(X_train.shape[1]))
    print('Size of output: {}'.format(1))