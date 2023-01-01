# generic imports
import os
import random
import pandas as pd
import numpy as np
import torch
import re
import ast

from mysutils.text import remove_urls

from arguments import getArguments

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


# preprocess 
def preprocess_urls(tokens):

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

    # convert strings to lists
    

    # Remove URLs
    train_df['tokens'] = train_df['tokens'].apply( preprocess_urls )
    val_df['tokens'] = val_df['tokens'].apply( preprocess_urls )

    return train_df, val_df