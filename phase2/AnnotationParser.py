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

