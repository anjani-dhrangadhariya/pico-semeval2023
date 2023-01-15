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

# lower-casing without altering the Abbreviations
from lower_caser import SmartLowercase
lower_caser =  SmartLowercase()

def transform(tokens):
    tokens_transformed = []
    for t_i in tokens:
        t_transformed = lower_caser(t_i)
        tokens_transformed.append( t_transformed )
    return tokens_transformed

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

picos_mapping = {'participant': 1, 'intervention':2, 'outcome':3, 'oos':0}

# get arguments
args = getArguments() # get all the experimental arguments

# process labels for an entity converts all the non-entity labels to 0's 
def process_labels(x):
    ent = args.entity

    if ent == 'all':
        # convert all the entities into class 1. 
        for counter, i in enumerate(x):
            if i > 0:
                x[counter] = 1
    elif ent == 'all_sep':
        # print( set(list(x)) )
        x = x
    else:
        for counter, i in enumerate(x):
            if i != picos_mapping[ent] and i != 0:
                x[counter] = 0
        for counter, i in enumerate(x):
            if i != 0:
                x[counter] = 1

    return x

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
                if len(replaced_text) == 0:
                    replaced_text = 'link'
                tokens[counter] = replaced_text

    return tokens

def preprocess_token_claim_offsets(x):

    dictionary = dict(zip(['N.A.', 'claim_starts', 'claim_ends'], [0, 1, 2]))

    return [dictionary[i] for i in x]

def fetch_val():

    train = 'st2_train_preprocessed.tsv'
    val = 'st2_val_preprocessed.tsv'

    train_df = pd.read_csv(f'{args.data_dir}/{train}', sep='\t')
    val_df = pd.read_csv(f'{args.data_dir}/{val}', sep='\t') 

    val_df['token_claim_offsets'] = val_df['token_claim_offsets'].apply( str_2_list )
    val_df['token_claim_offsets'] = val_df['token_claim_offsets'].apply( preprocess_token_claim_offsets )

    return val_df['token_claim_offsets']


# Load dataframe with PICO
def load_data(input_directory):

    train = 'st2_train_preprocessed.tsv'
    val = 'st2_val_preprocessed.tsv'
    test = 'st2_test_preprocessed.tsv'

    train_df = pd.read_csv(f'{input_directory}/{train}', sep='\t')
    val_df = pd.read_csv(f'{input_directory}/{val}', sep='\t') 
    test_df = pd.read_csv(f'{input_directory}/{test}', sep='\t') 

    # convert strings to Lists
    train_df['tokens'] = train_df['tokens'].apply( str_2_list )
    train_df['labels'] = train_df['labels'].apply( str_2_list )
    train_df['pos'] = train_df['pos'].apply( str_2_list )
    train_df['lemma'] = train_df['lemma'].apply( str_2_list )
    train_df['token_claim_offsets'] = train_df['token_claim_offsets'].apply( str_2_list )

    val_df['tokens'] = val_df['tokens'].apply( str_2_list )
    val_df['labels'] = val_df['labels'].apply( str_2_list )
    val_df['pos'] = val_df['pos'].apply( str_2_list )
    val_df['lemma'] = val_df['lemma'].apply( str_2_list )
    val_df['token_claim_offsets'] = val_df['token_claim_offsets'].apply( str_2_list )

    test_df['tokens'] = test_df['tokens'].apply( str_2_list )
    test_df['labels'] = test_df['labels'].apply( str_2_list )
    test_df['pos'] = test_df['pos'].apply( str_2_list )
    test_df['lemma'] = test_df['lemma'].apply( str_2_list )
    test_df['token_claim_offsets'] = test_df['token_claim_offsets'].apply( str_2_list )

    # smart lower casing
    train_df['tokens'] = train_df['tokens'].apply( transform )
    val_df['tokens'] = val_df['tokens'].apply( transform )
    test_df['tokens'] = test_df['tokens'].apply( transform )

    train_df['lemma'] = train_df['lemma'].apply( transform )
    val_df['lemma'] = val_df['lemma'].apply( transform )
    test_df['lemma'] = test_df['lemma'].apply( transform )

    # Remove URLs
    train_df['tokens'] = train_df['tokens'].apply( preprocess_urls )
    val_df['tokens'] = val_df['tokens'].apply( preprocess_urls )
    test_df['tokens'] = test_df['tokens'].apply( preprocess_urls )

    # Binarize POS tags
    train_df['pos'] = train_df['pos'].apply( pos_2_numeric )
    val_df['pos'] = val_df['pos'].apply( pos_2_numeric )
    test_df['pos'] = test_df['pos'].apply( pos_2_numeric )

    # select the entity
    train_df['labels'] = train_df['labels'].apply( process_labels )
    val_df['labels'] = val_df['labels'].apply( process_labels )
    test_df['labels'] = test_df['labels'].apply( process_labels )

    # token_claim_offsets to numbers
    train_df['token_claim_offsets'] = train_df['token_claim_offsets'].apply( preprocess_token_claim_offsets )
    val_df['token_claim_offsets'] = val_df['token_claim_offsets'].apply( preprocess_token_claim_offsets )
    test_df['token_claim_offsets'] = test_df['token_claim_offsets'].apply( preprocess_token_claim_offsets )

    return train_df, val_df, test_df