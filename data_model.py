
"""
This class abstracts the data.
"""

DEFAULT_PATH = "data/courses.txt"

class DataModel():

    def training_data(self):
        return self.training

    def testing_data(self):
        return self.testing

    def get_courses_way_training(self, way):
        return self.training


    def parse_data(self):
        with open(self.data_path, 'r') as f:
            

                

    def __init__(self, path=DEFAULT_PATH):
        self.data_path = path
        self.parse_data()

