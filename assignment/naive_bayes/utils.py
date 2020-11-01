from random import seed, randrange, shuffle
from math import sqrt, exp, pi
from csv import reader

def load_csv(filename): #U
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def train_test_split(dataset, train_size=0.8): 
  shuffle(dataset)
  X_train = dataset[:int(0.8*len(dataset))]
  X_test = dataset[int(0.8*len(dataset)):]
  return X_train, X_test

# Split a dataset into k folds 
def cross_validation_split(dataset, n_folds):
  dataset_split = list()
  l = len(dataset) // n_folds
  for i in range(n_folds):
    dataset_split.append(dataset[i*l:(i+1)*l])
  return dataset_split

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/max(1., float(len(numbers)))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	if len(numbers)==1:
		return 0
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset  
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column), mean(column)+3*stdev(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries


# Calculate the Gaussian probability distribution function for x 
def calculate_probability(x, mean, stdev):
	if stdev == 0: return 0
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
 