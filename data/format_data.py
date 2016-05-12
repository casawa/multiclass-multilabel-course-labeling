import os
import sys

data_file = 'courses.txt'
stop_words_file = 'english.stop'

# This script takes the raw data from the Explore Courses API and formats it
# to better work with our algorithms


def remove_stop_words():
    words = []
    with open(stop_words_file, 'r') as f:
        words = f.readlines()



def main():

    with open(data_file, 'r') as raw_data:
                


if __name__ == '__main__':
    main()
