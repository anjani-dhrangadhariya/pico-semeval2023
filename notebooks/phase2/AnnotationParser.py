# imports

import pandas as pd
import ast
import collections
import re
import spacy
import scispacy
import time
import numpy as np


#loading the scispacy model
nlp = spacy.load('en_core_sci_sm')


# open file
file_path_train = '/mnt/nas2/data/systematicReview/semeval2023/data/st2_train_inc_text.csv'
file_path_test = '/mnt/nas2/data/systematicReview/semeval2023/data/st2_test_inc_text.csv'

df_train = pd.read_csv(file_path_train, sep=',')
train_df = df_train.to_dict('records')

df_test = pd.read_csv(file_path_test, sep=',')
test_df = df_test.to_dict('records')

# Ignore the delted posts
deleted_list = ['[deleted]', '[removed]', 'deleted by user']


# Label mapping
picos_mapping = {'population': 1, 'intervention':2, 'outcome':3}


def get_char_labels(df):
    
    # parse dataframe and fetch annotations in a dict

    labels = []
    claim_offsets = []

    for counter, row in enumerate(df):
        #print('--------------------------------------------------------')

        reddit_id = row['subreddit_id']

        post_id = row['post_id']
        #print( post_id )

        claim = row['claim']
        #print('Claim: ', claim)

        full_text = row['text']
        #print('Full-Text: ', full_text)         

        if any(word in full_text for word in deleted_list):
            # If the post was removed by the user
            labels.append('N.A.')
            claim_offsets.append('N.A.')
        else:
            # If the post was not removed by the user
            # Get entities
            stage2_labels = ast.literal_eval( row['stage2_labels']  )
            #print( 'MAIN:     ', stage2_labels )
            stage2_labels = stage2_labels[0]['crowd-entity-annotation']['entities']
            #print( 'OFFSHOOT:     ', stage2_labels )

            # Get Char indices
            full_text_indices = [ counter for counter, i in enumerate(full_text) ]
            #print( 'full_text_indices:     ', full_text_indices )

            label_each_char = [0] * len(full_text) # Generate a 0 label for each character in the full text
            
            claim_start = full_text.index(claim)
            claim_end = claim_start + len(claim)

            for l in stage2_labels:
                extrct_annot = row['text'][ l['startOffset'] : l['endOffset'] ]
                pico =  l['label']

                # Are the start and stop offsets in the full-text offsets?
                if l['startOffset'] in full_text_indices:

                    prev_length = len(label_each_char)
                    start = l['startOffset']
                    end = l['startOffset']+(len(extrct_annot))
                    label_indices = [ i for i in range(start, end) ]

                    for i in range( start, end ):
                        old_label = label_each_char[i]
                        new_label = picos_mapping[pico]
                        if new_label > old_label:
                            label_each_char[i] = new_label
                    assert len(label_each_char) == prev_length
                    assert len(label_each_char) == len(full_text)
                    #print( label_each_char )

            labels.append(label_each_char)
            claim_offsets.append( (claim_start,claim_end) )
            
    return labels, claim_offsets


# Get the labels for train dataframe
labels_train, claim_offsets = get_char_labels(train_df)
df_train['labels_char'] = labels_train
df_train['claim_offsets'] = claim_offsets


def get_token_labels(df):
    
    tokens_series = []
    labels_series = []
    
    for counter, row in enumerate(df):
        #print('--------------------------------------------------------')

        reddit_id = row['subreddit_id']

        post_id = row['post_id']
        claim = row['claim']
        full_text = row['text']
        char_labels = row['labels_char']
        
        tokens = []
        token_labels = []


        if 'N.A.' not in char_labels:
            assert len(full_text) == len(char_labels)
            
            tokenized_text = [(m.group(0), m.start(), m.end() - 1) for m in re.finditer(r'\S+', full_text)]
            
            for counter, i in enumerate(tokenized_text):
                start = i[1]
                end = i[2] + 1 
                char_to_token_lab = list(set(char_labels[ start : end ]))
                if len(char_to_token_lab) == 1:
                    tokens.append( i[0] )
                    token_labels.append( char_to_token_lab[0] )

                else:
                    
                    #tokenize further
                    new_text = tokenized_text[counter][0]
                    new_labels = char_labels[ start : end ]
                    #print(new_text , ' : ', new_labels)
                    
                    v = np.array( new_labels )
                    tok_ind = np.where(np.roll(v,1)!=v)[0]
                    tok_ind = list(tok_ind)
                    if 0 not in tok_ind:
                        tok_ind = [0] + tok_ind

                    
                    new_text_tokens = [new_text[i:j] for i,j in zip(tok_ind, tok_ind[1:]+[None])]
                    new_text_labels = [new_labels[i:j] for i,j in zip(tok_ind, tok_ind[1:]+[None])]
                    
                    for t, l in zip(new_text_tokens, new_text_labels):
                        tokens.append( t )
                        token_labels.append( list(set(l))[0] )
                        
        else:
            tokens.append( ['N.A.'] )
            token_labels.append( ['N.A.'] )              
                        
        tokens_series.append(tokens)
        labels_series.append(token_labels)
                        
    return tokens_series, labels_series


# Get the labels for train dataframe
train_df = df_train.to_dict('records')
text_tokens, token_labels = get_token_labels(train_df)


df_train['tokens'] = text_tokens
df_train['labels'] = token_labels


# dump the dataframe to a csv file

write_parsed = '/mnt/nas2/data/systematicReview/semeval2023/data/parsed/st2_train_parsed.tsv'
#df_train.to_csv(write_parsed, encoding='utf-8', sep='\t')