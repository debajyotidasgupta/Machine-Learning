
"""
This python file contains the helper functions
related to the processing of the data
."""

# Authors: Debajyoti Dasgupta <debajyotidasgupta6@gmail.com>
#          Siba Smarak Panigrahi <sibasmarak.p@gmail.com>

import utils

def load_csv(filename):
	'''
	This function reads a .csv file and stores the data in a row-format (list of lists)
	Parameters:
	-----------
	filename: file path of .csv file to read

	Returns:
	--------
	dataset: data of .csv file in a list of list format
	'''

	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def get_column_index(items, cols): 
  	'''
  	This functions fetches the index of the column name specified in items from cols
  	Ex: items = ['B', 'C'] and cols = ['A', 'B', 'C', 'D']
  	    the function returns [1, 2]
  	Parameters:
  	-----------
  	items: list of column name(s) whose indices are required
  	cols: entire list of column names. This row was dropped after reading the data

  	Returns:
  	--------
  	items: list of intergers specifying the index of each of the original elements of items in cols
  	'''

	for i in range(len(cols)):
	    for j in range(len(items)):
	      	if cols[i]==items[j]:
	        	items[j]=i
	  	return items

def handle_missing_data(dataset, columns, missing_cols=['Bed Grade','City_Code_Patient']):
  	'''
  	This function imputes the missing value(s) of a feature with the most_frequent value of the feature
  	Parameters:
  	-----------
  	dataset: dataset which contains missing values and requires imputation
  	columns: original list of all features
  	missing_cols: list of features which contain missing_values
	
  	Returns:
  	--------
  	dataset: imputed dataset with no missing values
  	'''

  	for col in get_column_index(missing_cols, columns):

    	# obtain the most frequent value
    	mode = Counter([dataset[i][col] for i in range(len(dataset))]).most_common(1)[0][0]
    	for i in range(len(dataset)):
      		if dataset[i][col] == '':
        	# replace with the most frequent value
        	dataset[i][col] = mode
  	return dataset

def encoded_dataset(dataset, cols, 
                    categorical_cols=['Hospital_type_code','Hospital_region_code','Department',
                                      'Ward_Type','Ward_Facility_Code','Type of Admission','Severity of Illness','Age','Stay']):

  	'''
  	This function encodes the categorical columns in the dataset with LabelEncoder
  	Parameters:
  	-----------
  	dataset: dataset whose columns are to be encoded
  	columns: original list of all features
  	categorical_cols: list of categorical featues

  	Returns:
  	--------
  	dataset: updated dataset with label encoded categorical values  
  	'''

  	dataset = handle_missing_data(dataset, cols)
  	iter = get_column_index(categorical_cols,cols)
  	for col in iter:
    	le = LabelEncoder()
    	le.fit([dataset[i][col] for i in range(len(dataset))])
    	for i in range(len(dataset)):
      		dataset[i][col] = le.transform([dataset[i][col]])[0]
  	return dataset

def convert_to_float(dataset):
  	'''
  	This function converts each feature value (except the target values) into float
  	Parameters:
  	-----------
  	dataset: dataset whose entries are to be converted to float
  	         the entries should not give any exception while converting to float

  	Returns:
 	--------
  	dataset: updated dataset where each entry is a float type
  	'''

  	for j in range(len(dataset[0])-1):
  	  for i in range(len(dataset)):
    	  dataset[i][j] = float(dataset[i][j])
  	return dataset

def normalize(dataset):
  	'''
  	This function normalizes each column of the dataset (divides each column value with the corresponding mean value)
  	Parameters:
  	-----------
  	dataset: dataset whose columns are to be normalized

  	Returns:
  	--------
  	dataset: normalized dataset. Each columns of the dataset adds to 1
  	'''

  	for col in range(len(dataset[0])-1):
    	factor = sum([dataset[i][col] for i in range(len(dataset))])
    	for i in range(len(dataset)):
    	  dataset[i][col] /= factor
  	return dataset

def prepare(dataset, cols):
  	'''
  	This functions preprocesses the dataset and prepares for splitting and training purposes
  	It imputes the missing data, encodes the categorical values, converts each entry into float and normalizes each feature
  	Parameters:
  	-----------
  	dataset: dataset to be pre-processed
  	cols: list of all features

  	Returns:
  	--------
  	dataset: pre-processed dataset 
  	'''

  	dataset = handle_missing_data(dataset, cols)
  	dataset = encoded_dataset(dataset, cols)
  	dataset = convert_to_float(dataset)
  	dataset = normalize(dataset)
  	return dataset

def remove_outliers(dataset, outlier=1):
  	'''
  	This function removes those samples from dataset which contain the number of outlier features more than the parameter 'outlier'
  	Parameters:
  	-----------
  	dataset: dataset whose (outlier) samples are to be removed 
  	outlier: threshold number of outlier value. 
  	         A sample of dataset can be removed if it contains outlier features more this parameter

  	Returns:
  	--------
  	new_dataset: new dataset with outlier samples dropped
  	'''

  	summary = summarize_dataset(dataset)
  	new_dataset = []
  	for i in range(len(dataset)):
    	outlier_match = 0
    	for j in range(len(dataset[0])-1):
    	  if dataset[i][j] > summary[j][3]:
    	    outlier_match+=1
    	if outlier_match < outlier:
    	  new_dataset.append(dataset[i])
  	return new_dataset

def sequential_backward_selection(dataset, cols):
  	'''
  	This function uses sequential backward selection method to remove features
  	Parameters:
  	-----------
  	dataset: dataset whose features are to be dropped in accordance to feature selection method
  	cols: list of all features 

  	Returns:
  	--------
  	new_dataset: updated dataset with features selected by sequential backward selection method dropped 
  	cols: updated list of features. A subset of input parameter cols
  	all_max: the final accuracy of naive_bayes obtained on updated dataset and updated features 
  	'''

  	vectors = [[item for item in col] for col in zip(*dataset)]

  	all_max, cur_max = -1, -1
  	cur_label = -1
  	# maximum iteration of outer loop is the number of features in the dataset
  	for _ in range(18):
    	change = False
    	cur_max = -1
    	cur_label = -1

    	# iterate over all cols to find the column to be dropped
    	# drop the column which leads to largest increase in the accuracy
    	for i in range(len(cols)):
    	  	new_vectors = [[j for j in i] for i in vectors]
    	  	new_vectors.pop(i)
    	  	new_dataset = [[item for item in col] for col in zip(*new_vectors)]

      	score, _ = evaluate_algorithm(X_train, naive_bayes, 1)
      	if cur_max < score[0]:
        	cur_max = score[0]
        	cur_label = i

    	# check if the maximum accuracy has increased
    	# if True then drop the cur_label
    	if cur_max > all_max:
      		change = True
      		all_max = cur_max
      		cols.pop(cur_label)
      		vectors.pop(cur_label)

    	# if there is no more increase in accuracy by dropping any feature
    	# break from the loop - stop sequential backward selection method
    	if not change:
      		break

  	new_dataset = [[item for item in col] for col in zip(*vectors)]
  	return new_dataset, cols, all_max