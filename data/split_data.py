import random
import sys
import os

"""
This script splits the data into a training and test set randomly.
"""

train_percent = 0.9

data_file = 'new_formatted_courses.txt'
train_file = 'new_training_data.txt'
test_file = 'new_testing_data.txt'

# Writes the data
def write_data(data, dataset_file):

    with open(dataset_file, 'w') as f:
        for example in data:
            f.write(example[0])
            f.write(example[1])

def main():

    raw_data = []
    with open(data_file, 'r') as f:
        
        i = 1 
        pair = []
        for line in f:
            if i % 2 == 1:          # Odd lines are the classes
                pair.append(line)

            else:                   # Even lines are the WAYS that the course satsifies
                pair.append(line)
                raw_data.append(pair)
                pair = []
            i += 1

    random.shuffle(raw_data)
    training_data = raw_data[:int(len(raw_data)*train_percent)]
    testing_data = raw_data[int(len(raw_data)*train_percent): ]
  
    write_data(training_data, train_file)
    write_data(testing_data, test_file)

if __name__ == '__main__':
    main()

