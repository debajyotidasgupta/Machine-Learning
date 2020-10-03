"""
This python file contains the class for the construction of the dcision tree 
from the input dataset and then forms regression tree out of it using the ID3 
algorithm and Variance as an measure of impurity
."""

# Authors: Debajyoti Dasgupta <debajyotidasgupta6@gmail.com>
#          Siba Smarak Panigrahi <sibasmarak.p@gmail.com>

import .utility

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
        decision_tree_root: this is the root of the actual decision tree
                            which is to be pruned and the current node 
                            resides inside the tree rooted at this node
        
        cur_error:  cur_error stores the current minimum error that can 
                    be achieved by the decision tree rooted at decision 
                    tree root

        X_valid: Validation set of the data that is used for the pruning 

        Returns
        -------
        err: this is the current minimum error that the tree has achieved 
                till now

        '''
        if self.left==None and self.right==None:
            return 10**18
        if self.left!=None: cur_error = min(cur_error, self.left.prune(decision_tree_root, cur_error, X_valid))
        if self.right!=None: cur_error = min(cur_error, self.right.prune(decision_tree_root, cur_error, X_valid))

        # store the data of the children nodes in temporary variable
        temp_left = self.left
        temp_right = self.right 
        temp_attr = self.attr 
        self.remove_children()

        # calculate the error on the new decision tree
        err,_ = predict(decision_tree_root, X_valid)

        # if the error on the new decision tree increases then
        # restore the children of the current node
        if err > cur_error:
            self.restore(temp_attr, temp_left, temp_right)
        
        err = min(err, cur_error)
        return err

def variance(data):
    '''
    This function is a helper function which helps to calculate the 
    varince of the data that is given as the input    
    
    Parameters
    ----------
    data: array of numbers whose variance is to be calculated

    Returns
    -------
    var: This is the variance of the numbers that are  given as 
            input in data. Var = (X - mean)^2/N
    '''
    
    mean, var = 0, 0
    for i in data: mean+=i
    
    # calculate the mean
    mean /= len(data)
    for i in data: var+=(i-mean)**2
    # calculate the variance
    var /= len(data)  
    return var  

def good_attr(data, attr_list):
    best, best_attr, split, mse = -1, '', 0, 0
    random.shuffle(attr_list)
    for attr in attr_list:
        attr_data = [{
            'val': i[attr], 
            'Increase rate': i['Increase rate']
        } for i in data]

        local_best, local_val = -1, 0
        data_left, data_right = [], sorted([i['Increase rate'] for i in attr_data])
        data_var, data_len = variance(data_right), len(attr_data)
        left_len, right_len = 0, data_len
    
        for i in range(1, len(attr_data)):
            mid = (attr_data[i-1]['val'] + attr_data[i]['val']) // 2
            data_left.append(data_right.pop(0))
            left_len, right_len = left_len+1 , right_len-1

        left_var = variance(data_left)
        right_var = variance(data_right)

        gain = data_var - (left_len/data_len*left_var + right_len/data_len*right_var)
        if gain>local_best:
            local_best = gain
            local_val = mid
    
        if local_best>best:
            best = local_best
            best_attr = attr
            split = local_val
            mse = data_var

  return best_attr, split, mse

df = read_data()