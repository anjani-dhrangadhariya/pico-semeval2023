# imports

import pandas as pd
import ast
import collections
import re
import spacy
import scispacy

# Load tokenizers

#loading the scispacy model
nlp = spacy.load('en_core_sci_sm')

# open file
file_path_train = '/mnt/nas2/data/systematicReview/semeval2023/data/st2_train_inc_text.csv'
file_path_test = '/mnt/nas2/data/systematicReview/semeval2023/data/st2_test_inc_text.csv'

df_train = pd.read_csv(file_path_train, sep=',')
train_df = df_train.to_dict('records')

df_test = pd.read_csv(file_path_test, sep=',')
test_df = df_test.to_dict('records')


deleted_list = ['[deleted]', '[removed]', 'deleted by user']

def get_char_labels(df):
    
    # parse dataframe and fetch annotations in a dict

    labels = []

    for counter, row in enumerate(df):
        #print('--------------------------------------------------------')

        reddit_id = row['subreddit_id']

        post_id = row['post_id']

        claim = row['claim']

        full_text = row['text']

        if any(word in full_text for word in deleted_list):
            # If the post was removed by the user
            labels.append('N.A.')
        else:
            # If the post was not removed by the user
            # Get entities
            stage2_labels = ast.literal_eval( row['stage2_labels']  )
            stage2_labels = stage2_labels[0]['crowd-entity-annotation']['entities']

            # Get Char indices
            full_text_indices = [ counter for counter, i in enumerate(full_text) ]

            label_each_char = [0] * len(full_text)

            for l in stage2_labels:
                extrct_annot = row['text'][ l['startOffset'] : l['endOffset'] ]  

                # Are the start and stop offsets in the offsets?
                if l['startOffset'] in full_text_indices:

                    prev_length = len(label_each_char)
                    start = l['startOffset']
                    end = l['startOffset']+(len(extrct_annot))
                    label_indices = [ i for i in range(start, end) ]
                    for i in range( start, end ):
                        old_label = label_each_char[i]
                        new_label = 1
                        if new_label > old_label:
                            label_each_char[i] = new_label
                    assert len(label_each_char) == prev_length

            labels.append(label_each_char)

    # return 
    return labels