from collections import defaultdict

"""
This class provides an abstraction for the data and useful helper methods.
"""

DEFAULT_TRAIN_PATH = "data/training_data.txt"
DEFAULT_TEST_PATH = "data/testing_data.txt"

class DataModel:

    # Get training data
    def get_training_data(self):
        return self.training_data

    # Get testing data
    def get_testing_data(self):
        return self.testing_data

    # Find all courses that satisfy a particular WAY. Set train_flag to True for training data
    def query_by_way(self, way, train_flag):
        if train_flag:
            return self.train_ways_to_courses[way]
        return self.test_ways_to_courses[way]

    # Parses the training and testing data files
    def parse_data(self, data_path, train_flag):
        data = []
        with open(data_path, 'r') as f:
            i = 1 
            description = ""
            for line in f:
                if i % 2 == 1:          # Odd lines are the classes
                    description = line

                else:                   # Even lines are the WAYS that the courses satsifies
                    ways = line.split()
                    tokens = description.split()
                    for way in ways:
                        data.append([tokens, way])
                        if train_flag:
                            self.train_ways_to_courses[way].append(tokens)
                        else:
                            self.test_ways_to_courses[way].append(tokens)

                i += 1
 
        return data

    def get_list_of_ways(self):
        return ['WAY-A-II', 'WAY-AQR', 'WAY-CE', 'WAY-ED', 'WAY-ER', 'WAY-FR', 'WAY-SI', 'WAY-SMA']

    # Initializes the DataModel with data in convenient format
    def __init__(self, train_path=DEFAULT_TRAIN_PATH, test_path=DEFAULT_TEST_PATH):
        self.train_path = train_path
        self.test_path = test_path

        self.train_ways_to_courses = defaultdict(list)
        self.test_ways_to_courses = defaultdict(list)

        self.training_data = self.parse_data(self.train_path, True)
        self.testing_data = self.parse_data(self.test_path, False)
