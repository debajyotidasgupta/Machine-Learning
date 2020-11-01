
"""
This python file contains the helper functions
related to the processing of the data
."""

# Authors: Debajyoti Dasgupta <debajyotidasgupta6@gmail.com>
#          Siba Smarak Panigrahi <sibasmarak.p@gmail.com>

import utils
import time
from csv import reader
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from utils import summarize_dataset
from model import evaluate_algorithm, get_test_accuracy


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

def prepare(dataset, cols, start):
  print("\n ================ HANDLING MISSING DATA ============ \n")
  dataset = handle_missing_data(dataset, cols)
  print("Time elapsed  =  {} ms".format(time.time()-start))
  print("\n ================= ENCODING DATASET ================ \n")
  dataset = encoded_dataset(dataset, cols)
  print("Time elapsed  =  {} ms".format(time.time()-start))
  print("\n ================= FORMATTING DATA ================= \n")
  dataset = convert_to_float(dataset)
  print("Time elapsed  =  {} ms".format(time.time()-start))
  print("\n ============== NORMALIZING DATASET ================ \n")
  dataset = normalize(dataset)
  print("Time elapsed  =  {} ms".format(time.time()-start))
  print("\n ============= DATA PROCESSING FINISHED ============ \n\n")
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
def sequential_backward_selection(dataset, cols, algorithm, acc):
  vectors = [[item for item in col] for col in zip(*dataset)]
  removed = []

  all_max, cur_max = acc, -1
  cur_label = -1
  for loop in range(17):
    change = False
    cur_max = -1
    cur_label = -1
    print('\nChecking for the removal of feature {}'.format(loop+1))

    for i in range(len(cols)-1):
      new_vectors = [[j for j in i] for i in vectors]
      new_vectors.pop(i)
      new_dataset = [[item for item in col] for col in zip(*new_vectors)]

      score, _ = evaluate_algorithm(new_dataset, algorithm, 2)
      _mean = sum(score) / 2
      if cur_max < _mean:
        cur_max = _mean
        cur_label = i

    if cur_max > all_max:
      print('Improved Accuracy: {}'.format(cur_max))
      change = True
      all_max = cur_max
      removed.append(cols.pop(cur_label))
      print('Label -> {} <- Dropped'.format(removed[-1]))
      vectors.pop(cur_label)

    if not change:
      break
  
  print('No more feaures remaining to drop\n')
  new_dataset = [[item for item in col] for col in zip(*vectors)]
  return new_dataset, cols, removed, all_max