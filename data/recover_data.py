import sys
import string
from PorterStemmer import PorterStemmer


ORIG_FILE = 'new_courses.txt'
STEMMED_FILE = 'new_formatted_courses.txt'
STEM_TRAIN_FILE = 'new_training_data.txt' 
STEM_TEST_FILE = 'new_testing_data.txt' 


stop_words_file = 'english.stop'
way_prefix = 'WAY-'


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
def format_description(text, stop_words, stem=False):
    words = text.split()

    stemmer = PorterStemmer()
    non_stop_words = []
    for word in words:
        if word not in stop_words:      # Not a stop word, so lower, remove punctuation, and stem
            lowered_token = remove_punctuation(word).lower()
            
            if not stem:
                non_stop_words.append(lowered_token)
            else:
                non_stop_words.append(stemmer.stem(lowered_token))

    return ' '.join(non_stop_words)


def reformat_file(file_path, unstemming_map):

    data_file = open(file_path, 'r')
    data = data_file.readlines()
    data_file.close()

    stripped_data = [line.rstrip() for line in data]

    with open('unstemmed_' + file_path, 'w') as new_file:
        i = 1
        for line in stripped_data:
            if i % 2 == 0:
                new_file.write(line + '\n')
            else:
                new_file.write(unstemming_map[' '.join(line.split()[2:])] + '\n')

            i += 1


def main():
    
    stop_words = get_stop_words()

    stemmed_file = open(STEMMED_FILE, 'r')
    original_file = open(ORIG_FILE, 'r')

    stemmed = stemmed_file.readlines()[1::2]
    #ways = stemmed_file.readlines()[0::2]
    
    original = []
    for line in original_file.readlines():
        if not line.startswith(way_prefix):
            original.append(line.rstrip())

    stemmed_file.close()
    original_file.close()

    without_tags = [' '.join(stem_line.rstrip().split()[2:]) for stem_line in stemmed]

    for line in original:
        if 'modern queer' in line:
            print line

    orig_tags = [' '.join(orig_line.rstrip().split()[2:]) for orig_line in original]
    
    unstemming_map = {}
    for orig_desc in orig_tags:

        #if 'modern queer' in orig_desc:
        #print orig_desc

        stemmed_desc = ' '.join(format_description(orig_desc, stop_words, stem=True).split())
        unstemmed_desc = ' '.join(format_description(orig_desc, stop_words).split())
        unstemming_map[stemmed_desc] = unstemmed_desc


    reformat_file(STEM_TRAIN_FILE, unstemming_map)  
    reformat_file(STEM_TEST_FILE, unstemming_map)  

if __name__ == '__main__':
    main()

