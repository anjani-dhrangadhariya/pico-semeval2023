# generic imports
import ast
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from mysutils.text import remove_urls
from sklearn.preprocessing import Binarizer
from spacy.glossary import GLOSSARY
lookup_dict = GLOSSARY

from arguments import getArguments

# converts POS tags to numeric values
pos_2_num = dict( zip(lookup_dict.keys(), [*range(len(lookup_dict))]) )
def pos_2_numeric(pos_ent):
    return [pos_2_num[i] for i in pos_ent]

#  Seed everything
def seed_everything( seed ):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# get arguments
args = getArguments() # get all the experimental arguments

# convert strings to lists
def str_2_list(x):

    if isinstance(x, str):
        return ast.literal_eval(x)
    else:
        x

# preprocess URLs
def preprocess_urls(tokens):

    if type(tokens) == str:
        tokens = ast.literal_eval( tokens )

    # Scheme (HTTP, HTTPS, FTP and SFTP):
    regex = r'(?:(https?|s?ftp):\/\/)?'

    # Compile the ReGeX pattern
    find_urls_in_string = re.compile(regex, re.IGNORECASE)

    sentence = ' '.join( tokens )
    url = find_urls_in_string.search(sentence)

    if url is not None and url.group(0) is not None:
        for counter, t in enumerate(tokens):
            if 'https' in t:
                replaced_text = remove_urls(t)
                tokens[counter] = replaced_text

    return tokens

# Load dataframe with PICO
def load_data(input_directory):

    train = 'st2_train_preprocessed.tsv'
    val = 'st2_val_preprocessed.tsv'

    train_df = pd.read_csv(f'{input_directory}/{train}', sep='\t')
    val_df = pd.read_csv(f'{input_directory}/{val}', sep='\t') 

    # convert strings to Lists
    train_df['tokens'] = train_df['tokens'].apply( str_2_list )
    train_df['labels'] = train_df['labels'].apply( str_2_list )
    train_df['pos'] = train_df['pos'].apply( str_2_list )
    train_df['lemma'] = train_df['lemma'].apply( str_2_list )

    val_df['tokens'] = val_df['tokens'].apply( str_2_list )
    val_df['labels'] = val_df['labels'].apply( str_2_list )
    val_df['pos'] = val_df['pos'].apply( str_2_list )
    val_df['lemma'] = val_df['lemma'].apply( str_2_list )

    # Remove URLs
    train_df['tokens'] = train_df['tokens'].apply( preprocess_urls )
    val_df['tokens'] = val_df['tokens'].apply( preprocess_urls )

    # Binarize POS tags
    train_df['pos'] = train_df['pos'].apply( pos_2_numeric )
    val_df['pos'] = val_df['pos'].apply( pos_2_numeric )
    

    return train_df, val_df
