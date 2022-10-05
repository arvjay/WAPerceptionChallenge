# From @learncodebygaming on github
import os

def generate_negative_file_description():
    #Open the output file for writing. Will overwrite all existing data in there.
    with open('neg.txt', 'w') as f:
        #loop over all the filenames
        for filename in os.listdir('negative'):
            f.write('negative/' + filename + '\n');
generate_negative_file_description()