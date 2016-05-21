import sys
import os

FILE_NAME = 'formatted_courses.txt'

with open(FILE_NAME, 'r') as f:
    i = 0
    num_ways = 0
    for line in f:
        i += 1
        if i % 2 == 0:
            num_ways += len(line.split())

    print num_ways
