


import re
from string import punctuation
import numpy as np
from keras.preprocessing.sequence import pad_sequences

char_lookup_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'punct', 'num', ' ']

char_lookup_dict = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
char_lookup_dict.append( ' ' )
identity_mat = np.identity(len(char_lookup_dict))
max_length = 1000
word_max_len = 30

# converts characters to numeric values
char_2_num = dict( zip(char_lookup_dict, [*range(len(char_lookup_dict))]) )

def char_2_numeric(words):

    shape = ( len( words ), word_max_len, len( char_lookup_dict ) )
    words_representation = np.empty( shape , dtype=np.float32)

    for counter, word in enumerate(words):

        if len( word ) > 25:
            word = word[:30]

        char_word_rep = np.array([identity_mat[char_lookup_dict.index(char.lower())] for char in list(word) if char.lower() in char_lookup_dict], dtype=np.float32)
        char_word_rep_padded = np.concatenate( (char_word_rep, np.zeros((word_max_len - len(char_word_rep), len(char_lookup_dict)), dtype=np.float32)))

        char_word_rep_padded = char_word_rep_padded.reshape(1, *char_word_rep_padded.shape) 
        words_representation = np.append( words_representation, char_word_rep_padded, axis = 0)

    
    # Pad sequences to max_length
    if words_representation.shape[0] > max_length:
        words_representation = words_representation[:max_length]
    else:
        words_representation = np.concatenate( (words_representation, np.zeros( (max_length - len(words_representation), word_max_len, len(char_lookup_dict) ), dtype=np.float32)) )


    return words_representation


def char_2_ortho(words):

    shape = ( len( words ), word_max_len, 4 )
    words_ortho_representation = np.empty( shape , dtype=np.float32)

    for counter, word in enumerate(words):

        if len( word ) > 25:
            word = word[:30]



            

def transform_char(df):

    # get character (one-hot) encodings for tokens
    df['char_encode'] = df.tokens.apply( char_2_numeric )

    print( 'Fetched character encodings...' )
    df['char_ortho'] = df.tokens.apply( char_2_ortho )



    return None