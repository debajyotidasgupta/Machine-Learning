import data_processor
import utility

# Evaluate an algorithm using a cross validation split ==model
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
  folds = cross_validation_split(dataset, n_folds)
  scores = list()
  global_scores = -1
  optimal_summary = None
  for fold in folds:
    train_set = list(folds)
    train_set.remove(fold)
    train_set = sum(train_set, [])
    valid_set = list()
    for row in fold:
      row_copy = list(row)
      valid_set.append(row_copy)
      row_copy[-1] = None
    predicted, summary = algorithm(train_set, valid_set, *args)
    actual = [row[-1] for row in fold]
    accuracy = accuracy_metric(actual, predicted)
    
    if accuracy > global_scores:
      global_scores = accuracy
      optimal_summary = summary
    scores.append(accuracy)
  return scores, optimal_summary

def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Split dataset by class then calculate statistics for each row  ====model
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries


# Calculate the probabilities of predicting each class for a given row  ====model
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
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