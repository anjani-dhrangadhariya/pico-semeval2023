


import re
from string import punctuation


char_lookup_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'punct', 'num', ' ']

# converts characters to numeric values
char_2_num = dict( zip(char_lookup_dict, [*range(len(char_lookup_dict))]) )

def char_2_numeric(words):

    characterized_words = []

    for word in words:

        characterized = []

        # check if the word has a punctuation
        has_punct = any(p in word for p in punctuation)

        # check if the word has a number
        has_number = re.findall('[0-9]+', word)
        
        # check if the word has a space

        if len(has_number) > 0:
            for char in word:
                char = char.lower()
                if char in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    characterized.append( char_2_num['num'] ) 

        elif has_punct == True:
            for char in word:
                char = char.lower()
                if char in punctuation:
                    characterized.append( char_2_num['punct'] ) 

        else:
            for char in word:
                char = char.lower()
                characterized.append( char_2_num[char] ) 

        characterized_words.append( characterized )

    return characterized_words

def transform_char(df):

    # get character (one-hot) encodings for tokens
    df['char_encode'] = df.tokens.apply( char_2_numeric )

    print( 'Fetched character encodings...' )



    return None