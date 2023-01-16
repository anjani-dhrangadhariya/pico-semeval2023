


import re
from string import punctuation
import numpy as np
from keras.preprocessing.sequence import pad_sequences

char_lookup_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'punct', 'num', ' ']

char_lookup_dict = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
char_lookup_dict.append( ' ' )
identity_mat = np.identity( len(char_lookup_dict) )

ortho_lookup_dict = list( 'pncC' )
ortho_identity_mat = np.identity( len(ortho_lookup_dict) )

max_length = 512
word_max_len = 30

# converts characters to numeric values
char_2_num = dict( zip(char_lookup_dict, [*range(len(char_lookup_dict))]) )

def char_2_numeric(words):

    shape = ( len( words ), word_max_len, len( char_lookup_dict ) )
    words_representation = np.empty( shape , dtype=np.float32)

    for counter, word in enumerate(words):

        if len( word ) > word_max_len:
            word = word[:word_max_len]

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

def get_orthographic(word):

    all = []

    for c in word:
        ortho_feature = ''

        # check if the word has a punctuation
        has_punct = any(p in c for p in punctuation)
        # check if the word has a number
        has_number = re.findall('[0-9]+', c)

        if has_punct == True:
            ortho_feature = 'p'

        elif len(has_number) > 0:
            ortho_feature = 'n'

        else:
            if c.isupper() == True:
                ortho_feature = 'C'
            else:
                ortho_feature = 'c'

        all.append( ortho_feature )

    return ''.join(all)

def char_2_ortho(words):

    shape = ( len( words ), word_max_len, 4 )
    words_ortho_representation = np.empty( shape , dtype=np.float32)

    for counter, word in enumerate(words):

        if len( word ) > word_max_len:
            word = word[:word_max_len]

        word = get_orthographic(word)

        char_ortho_rep = np.array( [ortho_identity_mat[ ortho_lookup_dict.index(char.lower()) ] for char in list(word) if char.lower() in ortho_lookup_dict], dtype=np.float32)
        char_ortho_rep_padded = np.concatenate( (char_ortho_rep, np.zeros((word_max_len - len(char_ortho_rep), len(ortho_lookup_dict)), dtype=np.float32)))

        char_ortho_rep_padded = char_ortho_rep_padded.reshape(1, *char_ortho_rep_padded.shape) 
        words_ortho_representation = np.append( words_ortho_representation, char_ortho_rep_padded, axis = 0)

    # Pad sequences to max_length
    if words_ortho_representation.shape[0] > max_length:
        words_ortho_representation = words_ortho_representation[:max_length]
    else:
        words_ortho_representation = np.concatenate( (words_ortho_representation, np.zeros( (max_length - len(words_ortho_representation), word_max_len, len(ortho_lookup_dict) ), dtype=np.float32)) )

    return words_ortho_representation

def transform_char(df):

    df['char_encode'] = df.tokens.apply( char_2_numeric )
    df['char_ortho'] = df.tokens.apply( char_2_ortho )

    return df