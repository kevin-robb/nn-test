# This file has helper functions to handle importing data,
# transforming all attributes to float,
# changing the labels to integers,
# and normalizing all inputs to the range (0,1).

# from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from csv import reader
from random import randrange

def get_data():
    # load and prepare data
    filename = "data/wheat-seeds.csv"
    dataset = load_csv(filename)
    # turn all input columns to floats
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    return dataset

## data entry/management helper functions ---------------------------------------------

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Calculate accuracy percentage
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i][-1] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
