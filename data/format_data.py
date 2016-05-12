import os
import sys
import string

data_file = 'courses.txt'
stop_words_file = 'english.stop'
way_prefix = 'WAY-'

# This script takes the raw data from the Explore Courses API and formats it
# to better work with our algorithms


# Gets stop words
def get_stop_words():
    words = []
    with open(stop_words_file, 'r') as f:
        for line in f:
            words.append(line.rstrip())
    
    return frozenset(words)

# Removes punctuation
def remove_punctuation(word):
    return word.translate(None, string.punctuation)

# Formats the description by removing stop words and punctuation
def format_description(text, stop_words):
    words = text.split()

    non_stop_words = []
    for word in words:
        if word not in stop_words:      # Not a stop word, so lower and remove punctuation
            non_stop_words.append(remove_punctuation(word).lower())

    return ' '.join(non_stop_words)

# Reads through the data file and formats it for better and more efficient parsing by Data Model
def main():

    stop_words = get_stop_words()

    with open(data_file, 'r') as raw_data:
        ways = []
        considered_courses = set()
        for line in raw_data:
            line = line.rstrip()
            if line.startswith(way_prefix):
                ways.append(line)
            else:
                if line not in considered_courses:
                    considered_courses.add(line)
                    line = format_description(line, stop_words)
                    print line
                    print ' '.join(ways) 

                ways = []


if __name__ == '__main__':
    main()
