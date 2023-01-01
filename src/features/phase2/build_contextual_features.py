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


##################################################################################
# The function truncates input sequences to max lengths
##################################################################################
def truncateSentence(sentence, trim_len):

    trimmedSentence = []
    if  len(sentence) > trim_len:
        trimmedSentence = sentence[:trim_len]
    else:
        trimmedSentence = sentence

    assert len(trimmedSentence) <= trim_len
    return trimmedSentence


##################################################################################
# The function adds special tokens to the truncated sequences
##################################################################################
def addSpecialtokens(eachText, start_token, end_token):
    insert_at_start = 0
    eachText[insert_at_start:insert_at_start] = [start_token]

    insert_at_end = len(eachText)
    eachText[insert_at_end:insert_at_end] = [end_token]

    assert eachText[0] == start_token
    assert eachText[-1] == end_token

    return eachText


def transform(df, tokenizer, max_length, pretrained_model):

    tokenized = []
    for tokens, labels, pos, lemma in zip(list(df['tokens']), list(df['labels']), list(df['pos']), list(df['lemma'])) :

        # Tokenize and preserve labels
        tokens_ = ast.literal_eval( tokens )
        labels_ = ast.literal_eval( labels )
        pos_ = ast.literal_eval( pos )
        lemma_ = ast.literal_eval( lemma )

        tok_sentence, tok_labels, tok_pos, tok_lemma = tokenize_and_preserve_labels(tokens_, labels_, pos_, lemma_, tokenizer)

        # Truncate the sequences (sentence and label) to (max_length - 2)
        if max_length >= 510:
            tokens_trunc = truncateSentence(tok_sentence, (max_length - 2))
            labels_trunc = truncateSentence(tok_labels, (max_length - 2))
            pos_trunc = truncateSentence(tok_pos, (max_length - 2))
            lemma_trunc = truncateSentence(tok_lemma, (max_length - 2))
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc) == len(lemma_trunc)
        else:
            tokens_trunc = tok_sentence
            labels_trunc = tok_labels
            pos_trunc = tok_pos
            lemma_trunc = tok_lemma
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc) == len(lemma_trunc)


        # Add special tokens CLS and SEP for the BERT tokenizer (identical for SCIBERT)
        if 'bert' in pretrained_model.lower():
            tokens_spetok = addSpecialtokens(tokens_trunc, tokenizer.cls_token_id, tokenizer.sep_token_id)
        elif 'gpt2' in pretrained_model.lower():
            tokens_spetok = addSpecialtokens(tokens_trunc, tokenizer.bos_token_id, tokenizer.eos_token_id)

        if any(isinstance(i, list) for i in labels_trunc) == False:
            labels_spetok = addSpecialtokens(labels_trunc, 0, 0)
        else:
            labels_spetok = [[0.0,0.0]] + labels_trunc + [[0.0,0.0]]

        pos_spetok = addSpecialtokens(pos_trunc, 0, 0)
        lemma_spetok = addSpecialtokens(lemma_trunc, 0, 0)

        print( tokens_[-1], tok_sentence[-1], tokens_trunc[-1], tokens_spetok[-1] )
