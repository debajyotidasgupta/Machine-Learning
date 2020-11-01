"""
This python file contains the functions for  the
implementation of the naive bayes algorithm from 
the input dataset and then return optimal  model

The file also contains the model helper functions
for  proper  implementation  of  the  naive bayes 
algorithm
"""

# Authors: Debajyoti Dasgupta <debajyotidasgupta6@gmail.com>
#          Siba Smarak Panigrahi <sibasmarak.p@gmail.com>


from utils import cross_validation_split, accuracy_metric, summarize_dataset, calculate_probability

# Evaluate an algorithm using a cross validation split ==model
def evaluate_algorithm(dataset, algorithm, n_folds):
  '''
  This function is the main evaluation function
  which collects together all the steps that need
  to be performed in the naive bayes classifier

  Parameter
  ---------
  dataset: the dataset over which naive bayes classifier
          should be trained
  
  algorithms: The  function  handling  the  naive  bayes 
              algorithm

  n_folds: number  of  folds  for  the cross validaation

  Return:
  -------
  scores: The accuracies achieve in each of the n_fold 
          cross validation steps
  
  optimal summary: The model parameters for the optimal
                    model achieved in cross  validation
  '''
  # get the split dataset for the cross validation
  folds = cross_validation_split(dataset, n_folds)
  scores = list()
  global_scores = -1
  optimal_summary = None

  # iterate over each fold one at a time and make it he validation set
  for fold in folds:
    train_set = list(folds)
    train_set.remove(fold)
    train_set = sum(train_set, [])
    valid_set = list()
    for row in fold:
      row_copy = list(row)
      valid_set.append(row_copy)
      row_copy[-1] = None

    # Run the Naive bayes Algorithm to get the predictions
    predicted, summary = algorithm(train_set, valid_set)
    actual = [row[-1] for row in fold]
    accuracy = accuracy_metric(actual, predicted)
    
    # if there is an improvement in accuracy then select this model
    if accuracy > global_scores:
      global_scores = accuracy
      optimal_summary = summary

    # append the accuracy obtained in this iteration in the list of scores
    scores.append(accuracy)
  return scores, optimal_summary

def separate_by_class(dataset):
  '''
  This function is used to split  the  input 
  dataset as per the classes defined by  the 
  target class. The target class will be the 
  last column of the dataset

  Parameter
  ---------
  dataset: the dataset ove which the split has to
           be performed

  Return
  ------
  split_by_class: A dictionary where keys are   the 
                  classes of the target feature and 
                  the items are the  dataset  which 
                  have the key as their target class
  '''
  split_by_class = dict()
  for i in range(len(dataset)):
    vector = dataset[i]
    class_value = vector[-1]
    if (class_value not in split_by_class):
      split_by_class[class_value] = list()
    # inset the datapoint in the respective class
    split_by_class[class_value].append(vector)
  return split_by_class

def summarize_by_class(dataset):
  '''
  This function is used to get the summaries
  (mean, std dev, # of elem, threshhold) for 
  each feature corresponding to  each  class
  as determined by the target feature

  Parameter
  ---------
  dataset: Dataset over which the summarization
           needs to be performed
  
  Return
  ------
  summaries: the  summaries  corresponding  to 
             each feature ordered by the class
             they belong to. (class -> keys)
  '''
  separated = separate_by_class(dataset)
  summaries = dict()
  for class_value, rows in separated.items():
    summaries[class_value] = summarize_dataset(rows)
  return summaries


def calculate_class_probabilities(summaries, data):
  '''
  Helper function to calculate the probabilities of
  each class given  the  current  data  value. This 
  function helps in calculating the aposteriori for
  each class

  Parameter
  ---------
  summaries: the summaries of each feature calculated 
             while  training  the  model on the train 
             dataset

  data: A unit data vector, for which the class 
        probabilities need to be calculated

  Return
  ------
  
  '''
  total_rows = sum([summaries[label][0][2] for label in summaries])
  probabilities = dict()
  for class_value, class_summaries in summaries.items():
    probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
    for i in range(len(class_summaries)):
      mean, stdev, _, _ = class_summaries[i]
      probabilities[class_value] *= calculate_probability(data[i], mean, stdev)
  return probabilities


# Predict the class for a given row  model
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Naive Bayes Algorithm   model
def naive_bayes(train, test):
  summarize = summarize_by_class(train)
  predictions = []
  for row in test:
    output = predict(summarize, row)
    predictions.append(output)
  return (predictions), summarize

  #model
def get_test_accuracy(test, summarize):
  predictions = []
  for row in test:
    output = predict(summarize, row)
    predictions.append(output)
  actual = [row[-1] for row in test]
  accuracy = accuracy_metric(actual, predictions)
  return accuracy