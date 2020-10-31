
"""
This python file contains the helper functions
related to the processing of the data
."""

# Authors: Debajyoti Dasgupta <debajyotidasgupta6@gmail.com>
#          Siba Smarak Panigrahi <sibasmarak.p@gmail.com>

import utils

def load_csv(filename): #U
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# get the column index of the items
def get_column_index(items, cols): #U
  for i in range(len(cols)):
    for j in range(len(items)):
      if cols[i]==items[j]:
        items[j]=i
  return items

#replace with most frequent
def handle_missing_data(dataset, columns):
  for col in get_column_index(['Bed Grade','City_Code_Patient'], columns):
    mode = Counter([dataset[i][col] for i in range(len(dataset))]).most_common(1)[0][0]
    for i in range(len(dataset)):
      if dataset[i][col] == '':
        dataset[i][col] = mode
  return dataset

def encoded_dataset(dataset, cols):
  dataset = handle_missing_data(dataset, cols)
  iter = get_column_index([
              'Hospital_type_code',
              'Hospital_region_code',
              'Department',
              'Ward_Type',
              'Ward_Facility_Code',
              'Type of Admission',
              'Severity of Illness',
              'Age',
              'Stay'
             ],cols)
  for col in iter:
    le = LabelEncoder()
    le.fit([dataset[i][col] for i in range(len(dataset))])
    for i in range(len(dataset)):
      dataset[i][col] = le.transform([dataset[i][col]])[0]
  return dataset

def convert_to_float(dataset):
  for j in range(len(dataset[0])-1):
    for i in range(len(dataset)):
      dataset[i][j] = float(dataset[i][j])
  return dataset

def normalize(dataset):
  for col in range(len(dataset[0])-1):
    factor = sum([dataset[i][col] for i in range(len(dataset))])
    for i in range(len(dataset)):
      dataset[i][col] /= factor
  return dataset

def prepare(dataset, cols):
  dataset = handle_missing_data(dataset, cols)
  dataset = encoded_dataset(dataset, cols)
  dataset = convert_to_float(dataset)
  dataset = normalize(dataset)
  return dataset

# data_processor
def remove_outliers(dataset, outlier=1):
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

#data_processor
def sequential_backward_selection(dataset, cols):
  vectors = [[item for item in col] for col in zip(*dataset)]

  all_max, cur_max = -1, -1
  cur_label = -1
  for _ in range(18):
    change = False
    cur_max = -1
    cur_label = -1

    for i in range(len(cols)):
      new_vectors = [[j for j in i] for i in vectors]
      new_vectors.pop(i)
      new_dataset = [[item for item in col] for col in zip(*new_vectors)]

      score, _ = evaluate_algorithm(X_train, naive_bayes, 1)
      if cur_max < score[0]:
        cur_max = score[0]
        cur_label = i

    if cur_max > all_max:
      change = True
      all_max = cur_max
      cols.pop(cur_label)
      vectors.pop(cur_label)

    if not change:
      break

  new_dataset = [[item for item in col] for col in zip(*vectors)]
  return new_dataset, cols, all_max