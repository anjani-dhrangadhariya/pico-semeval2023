import ast
from arguments import getArguments

from choose_embed_type import choose_tokenizer_type

# load arguments
args = getArguments() # get all the experimental arguments

# load tokenizer
tokenizer, _ = choose_tokenizer_type( args.embed )

def tokenize_and_preserve_labels(sentence, text_labels, pos, lemma, tokenizer):

    tokenized_sentence = []
    labels = []
    poss = []
    lemmas = []

    for word, label, pos_i, lemma_i in zip(sentence, text_labels, pos, lemma):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.encode(word, add_special_tokens = False)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        if n_subwords == 1:
            labels.extend([label] * n_subwords)
            poss.extend( [pos_i] * n_subwords )
            lemmas.extend( [lemma_i] * n_subwords )
        elif n_subwords == 0:
            pass
        else:

            if isinstance( label, list ) == False:
                
                dummy_label = 100

                labels.extend([label])
                labels.extend( [dummy_label] * (n_subwords-1) )
                poss.extend( [pos_i] * n_subwords )
                lemmas.extend( [lemma_i] * n_subwords )
            else:

                dummy_label = [100.00, 100.00]

                labels.extend([label])
                labels.extend( [dummy_label] * (n_subwords-1) )
                poss.extend( [pos_i] * n_subwords )    
                lemmas.extend( [lemma_i] * n_subwords )            

    assert len(tokenized_sentence) == len(labels) == len(poss)

    return tokenized_sentence, labels, poss, lemmas


def transform(df, tokenizer, max_length, pretrained_model):

    tokenized = []
    for tokens, labels, pos, lemma in zip(list(df['tokens']), list(df['labels']), list(df['pos']), list(df['lemma'])) :

        # Tokenize and preserve labels
        tokens_ = ast.literal_eval( tokens )
        labels_ = ast.literal_eval( labels )
        pos_ = ast.literal_eval( pos )
        lemma_ = ast.literal_eval( lemma )

        tok_sentence, tok_labels, tok_pos, tok_lemma = tokenize_and_preserve_labels(tokens_, labels_, pos_, lemma_, tokenizer)

        
