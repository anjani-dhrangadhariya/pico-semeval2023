import ast
from arguments import getArguments
import numpy as np
from choose_embed_type import choose_tokenizer_type
from keras.preprocessing.sequence import pad_sequences


# load arguments
args = getArguments() # get all the experimental arguments

def tokenize_and_preserve_labels(sentence, text_labels, pos, claim_offset, tokenizer, text_labels_fine = None):

    if len(claim_offset) != len(sentence):
        claim_offset = claim_offset * len(sentence)

    tokenized_sentence = []
    labels = []
    labels_fine = []
    poss = []
    claim_offsets = []

    # if 'mtl' not in args.model:
    text_labels_fine = text_labels

    for word, label, label_fine, pos_i, offset_i in zip(sentence, text_labels, text_labels_fine, pos, claim_offset):

        # print( word )

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.encode(word, add_special_tokens = False)

        # print( tokenized_word )
        # print( tokenizer.convert_ids_to_tokens(tokenized_word) )

        n_subwords = len(tokenized_word)
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        if n_subwords == 1:
            labels.extend([label] * n_subwords)
            labels_fine.extend( [label_fine] * n_subwords )

            poss.extend( [pos_i] * n_subwords )
            claim_offsets.extend( [offset_i] * n_subwords )
        else:                
            dummy_label = 100

            labels.extend([label])
            labels.extend( [dummy_label] * (n_subwords-1) ) # switch from label to dummy label
            labels_fine.extend([label_fine])
            labels_fine.extend( [dummy_label] * (n_subwords-1) ) # switch from label to dummy label
            poss.extend( [pos_i] * n_subwords )
            claim_offsets.extend( [offset_i] * n_subwords ) 

    assert len(tokenized_sentence) == len(labels) == len(poss) == len(claim_offsets) 

    if 'mtl' not in args.model:
        return tokenized_sentence, labels, poss, claim_offsets
    else:
        return tokenized_sentence, labels, labels_fine, poss, claim_offsets



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
    claim_offset_trans = []

    if args.use_lemma == True:
        tokens = list(df['lemma'])
    else:
        tokens = list(df['tokens'])

    for tokens, labels, pos, claim_offset in zip( tokens, list(df['labels']), list(df['pos']), list(df['token_claim_offsets']) ) :

        # Tokenize and preserve labels
        if isinstance(tokens, str):
            tokens_ = ast.literal_eval( tokens )
            labels_ = ast.literal_eval( labels )
            pos_ = ast.literal_eval( pos )
            claim_offset_ = ast.literal_eval( claim_offset )
        else:
            tokens_ = tokens
            labels_ = labels
            pos_ = pos
            claim_offset_ = claim_offset

        # if len(tokens_) < 10:
        #     print( 'Type of tokens:', type(tokens_) )
        #     print( tokens_ )
        tok_sentence, tok_labels, tok_pos, tok_claim_offset = tokenize_and_preserve_labels(tokens_, labels_, pos_, claim_offset_, tokenizer)

        # if len( tokens_ ) < 40:
        #     print( tokens_ )
        #     print( tok_sentence )
        #     print( tokenizer.convert_ids_to_tokens(tok_sentence) )
    
        # Truncate the sequences (sentence and label) to (max_length - 2)
        if max_length >= 510:
            
            tokens_trunc = truncateSentence(tok_sentence, (max_length - 2))
            labels_trunc = truncateSentence(tok_labels, (max_length - 2))
            pos_trunc = truncateSentence(tok_pos, (max_length - 2))
            claim_offset_trunc = truncateSentence(tok_claim_offset, (max_length - 2))
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc)
        else:
            tokens_trunc = tok_sentence
            labels_trunc = tok_labels
            pos_trunc = tok_pos
            claim_offset_trunc = tok_claim_offset
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc) == len(tok_claim_offset)


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
        claim_offset_spetok = addSpecialtokens(claim_offset_trunc, 0, 0)


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

        input_claim_offset = pad_sequences([ claim_offset_spetok ] , maxlen=max_length, value=0, padding="post")
        input_claim_offset = input_claim_offset[0]

        assert len( input_ids ) == len( input_labels ) == len( input_pos ) == len( input_claim_offset )

        # Get the attention masks
        # TODO: Also mask the input ids that have labels [0.5,0.5]
        attention_masks = createAttnMask( [input_ids], [input_labels] ) 

        assert len(input_claim_offset.squeeze()) == len(input_ids.squeeze()) == len(input_labels.squeeze()) == len(attention_masks.squeeze()) == len(input_pos.squeeze()) == max_length

        tokens_trans.append( input_ids.squeeze() ) 
        labels_trans.append( input_labels.squeeze() ) 
        pos_trans.append(  input_pos.squeeze() ) 
        masks_trans.append( attention_masks.squeeze() )  
        claim_offset_trans.append( input_claim_offset.squeeze() ) 

    return tokens_trans, labels_trans, pos_trans, masks_trans, claim_offset_trans

def transform_mtl( df, tokenizer, max_length, pretrained_model, args ):

    tokens_trans = []
    labels_trans = []
    labels_fine_trans = []
    pos_trans = []
    masks_trans = []
    claim_offset_trans = []

    if args.use_lemma == True:
        tokens = list(df['lemma'])
    else:
        tokens = list(df['tokens'])

    for tokens, labels_coarse, labels_fine, pos, claim_offset in zip( tokens, list(df['labels_coarse']), list(df['labels_fine']), list(df['pos']), list(df['token_claim_offsets']) ) :

        # Tokenize and preserve labels
        if isinstance(tokens, str):
            tokens_ = ast.literal_eval( tokens )
            labels_coarse_ = ast.literal_eval( labels_coarse )
            labels_fine_ = ast.literal_eval( labels_fine )
            pos_ = ast.literal_eval( pos )
            claim_offset_ = ast.literal_eval( claim_offset )
        else:
            tokens_ = tokens
            labels_coarse_ = labels_coarse
            labels_fine_ = labels_fine
            pos_ = pos
            claim_offset_ = claim_offset

        tok_sentence, tok_labels, tok_labels_fine, tok_pos, tok_claim_offset = tokenize_and_preserve_labels(tokens_, labels_coarse_, pos_, claim_offset_, tokenizer, labels_fine_)

        # Truncate the sequences (sentence and label) to (max_length - 2)
        if max_length >= 510:
            tokens_trunc = truncateSentence(tok_sentence, (max_length - 2))
            labels_trunc = truncateSentence(tok_labels, (max_length - 2))
            labels_fine_trunc = truncateSentence(tok_labels_fine, (max_length - 2))
            pos_trunc = truncateSentence(tok_pos, (max_length - 2))
            claim_offset_trunc = truncateSentence(tok_claim_offset, (max_length - 2))
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc)
        else:
            tokens_trunc = tok_sentence
            labels_trunc = tok_labels
            labels_fine_trunc = tok_labels_fine
            pos_trunc = tok_pos
            claim_offset_trunc = tok_claim_offset
            assert len(tokens_trunc) == len(labels_trunc) == len(pos_trunc) == len(tok_claim_offset)


        # Add special tokens CLS and SEP for the BERT tokenizer (identical for SCIBERT)
        if 'bert' in pretrained_model.lower():
            tokens_spetok = addSpecialtokens(tokens_trunc, tokenizer.cls_token_id, tokenizer.sep_token_id)
        elif 'gpt2' in pretrained_model.lower():
            tokens_spetok = addSpecialtokens(tokens_trunc, tokenizer.bos_token_id, tokenizer.eos_token_id)

        if any(isinstance(i, list) for i in labels_trunc) == False:
            labels_spetok = addSpecialtokens(labels_trunc, 0, 0)
            labels_fine_spetok = addSpecialtokens(labels_fine_trunc, 0, 0)
        else:
            labels_spetok = [[0.0,0.0]] + labels_trunc + [[0.0,0.0]]
            labels_fine_spetok = [[0.0,0.0]] + labels_fine_trunc + [[0.0,0.0]]

        pos_spetok = addSpecialtokens(pos_trunc, 0, 0)
        claim_offset_spetok = addSpecialtokens(claim_offset_trunc, 0, 0)


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

            input_labels_fine = pad_sequences([ labels_fine_spetok ] , maxlen=max_length, value=0, padding="post")
            input_labels_fine = input_labels_fine[0]
        else:
            padding_length = max_length - len(labels_spetok)

            padding = [ [0.0,0.0] ]  * padding_length
            input_labels = labels_spetok + padding
            # Change dtype of list here
            input_labels = np.array( input_labels )

        input_pos = pad_sequences([ pos_spetok ] , maxlen=max_length, value=0, padding="post")
        input_pos = input_pos[0]

        input_claim_offset = pad_sequences([ claim_offset_spetok ] , maxlen=max_length, value=0, padding="post")
        input_claim_offset = input_claim_offset[0]

        assert len( input_ids ) == len( input_labels ) == len( input_labels_fine ) == len( input_pos ) == len( input_claim_offset )

        # Get the attention masks
        # TODO: Also mask the input ids that have labels [0.5,0.5]
        attention_masks = createAttnMask( [input_ids], [input_labels] ) 

        assert len(input_claim_offset.squeeze()) == len(input_ids.squeeze()) == len(input_labels.squeeze()) == len(attention_masks.squeeze()) == len(input_pos.squeeze()) == max_length

        tokens_trans.append( input_ids.squeeze() ) 
        labels_trans.append( input_labels.squeeze() ) 
        labels_fine_trans.append( input_labels_fine.squeeze() ) 
        pos_trans.append(  input_pos.squeeze() ) 
        masks_trans.append( attention_masks.squeeze() )  
        claim_offset_trans.append( input_claim_offset.squeeze() ) 

    return tokens_trans, labels_trans, labels_fine_trans, pos_trans, masks_trans, claim_offset_trans