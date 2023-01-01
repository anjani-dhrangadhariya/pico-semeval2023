import ast
from arguments import getArguments
import numpy as np
from choose_embed_type import choose_tokenizer_type
from keras.preprocessing.sequence import pad_sequences


# load arguments
args = getArguments() # get all the experimental arguments

def tokenize_and_preserve_labels(sentence, text_labels, pos, tokenizer):

    tokenized_sentence = []
    labels = []
    poss = []
    lemmas = []

    for word, label, pos_i in zip(sentence, text_labels, pos):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.encode(word, add_special_tokens = False)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        if n_subwords == 1:
            labels.extend([label] * n_subwords)
            poss.extend( [pos_i] * n_subwords )
        elif n_subwords == 0:
            pass
        else:

            if isinstance( label, list ) == False:
                
                dummy_label = 100

                labels.extend([label])
                labels.extend( [dummy_label] * (n_subwords-1) )
                poss.extend( [pos_i] * n_subwords )
            else:

                dummy_label = [100.00, 100.00]

                labels.extend([label])
                labels.extend( [dummy_label] * (n_subwords-1) )
                poss.extend( [pos_i] * n_subwords )    

    assert len(tokenized_sentence) == len(labels) == len(poss)

    return tokenized_sentence, labels, poss


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

##################################################################################
# Generates attention masks
##################################################################################
def createAttnMask(input_ids, input_lbs):

    # Mask the abstains
    # if isinstance(labels[0], int) == False:
    #     labels = mask_abstains(labels)

    # Add attention masks
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent, lab in zip(input_ids, input_lbs):
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [ int(token_id > 0) for token_id in sent ]

        if isinstance(lab[0], np.ndarray):
            for counter, l in enumerate(lab):
                if len(set(l)) == 1 and list(set(l))[0] == 0.5:
                    att_mask[counter] = 0
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return np.asarray(attention_masks, dtype=np.uint8)


def transform(df, tokenizer, max_length, pretrained_model, args):

    tokens_trans = []
    labels_trans = []
    pos_trans = []
    masks_trans = []

    if args.use_lemma == True:
        tokens = list(df['lemma'])
    else:
        tokens = list(df['tokens'])

    for tokens, labels, pos in zip( tokens, list(df['labels']), list(df['pos']) ) :

        # Tokenize and preserve labels
        if isinstance(tokens, str):
            tokens_ = ast.literal_eval( tokens )
            labels_ = ast.literal_eval( labels )
            pos_ = ast.literal_eval( pos )
        else:
            tokens_ = tokens
            labels_ = labels
            pos_ = pos

        tok_sentence, tok_labels, tok_pos = tokenize_and_preserve_labels(tokens_, labels_, pos_, tokenizer)

        # Truncate the sequences (sentence and label) to (max_length - 2)
        if max_length >= 510:
            tokens_trunc = truncateSentence(tok_sentence, (max_length - 2))
            labels_trunc = truncateSentence(tok_labels, (max_length - 2))
            pos_trunc = truncateSentence(tok_pos, (max_length - 2))
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc)
        else:
            tokens_trunc = tok_sentence
            labels_trunc = tok_labels
            pos_trunc = tok_pos
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc)


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


        # PAD the sequences to max length
        if 'bert' in pretrained_model.lower():
            input_ids = pad_sequences([ tokens_spetok ] , maxlen=max_length, value=tokenizer.pad_token_id, padding="post")
            input_ids = input_ids[0]
        elif 'gpt2' in pretrained_model.lower():
            input_ids = pad_sequences([ tokens_spetok ] , maxlen=max_length, value=tokenizer.unk_token_id, padding="post") 
            input_ids = input_ids[0]

        if any(isinstance(i, list) for i in labels_spetok) == False:
            input_labels = pad_sequences([ labels_spetok ] , maxlen=max_length, value=0, padding="post")
            input_labels = input_labels[0]
        else:
            padding_length = max_length - len(labels_spetok)
            padding = [ [0.0,0.0] ]  * padding_length
            input_labels = labels_spetok + padding
            # Change dtype of list here

            input_labels = np.array( input_labels )

        input_pos = pad_sequences([ pos_spetok ] , maxlen=max_length, value=0, padding="post")
        input_pos = input_pos[0]


        assert len( input_ids ) == len( input_labels ) == len( input_pos )


        # Get the attention masks
        # TODO: Also mask the input ids that have labels [0.5,0.5]
        attention_masks = createAttnMask( [input_ids], [input_labels] ) 

        assert len(input_ids.squeeze()) == len(input_labels.squeeze()) == len(attention_masks.squeeze()) == len(input_pos.squeeze()) == max_length

        tokens_trans.append( input_ids.squeeze() ) 
        labels_trans.append( input_labels.squeeze() ) 
        pos_trans.append( attention_masks.squeeze() ) 
        masks_trans.append( input_pos.squeeze() ) 


    return tokens_trans, labels_trans, pos_trans, masks_trans