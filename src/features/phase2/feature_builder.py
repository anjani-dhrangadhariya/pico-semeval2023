# generic imports
import ast
import sys
import time
from build_character_features import transform_char
from build_contextual_features import transform, transform_mtl

from tqdm.notebook import tqdm
import pandas as pd
from arguments import getArguments
from build_contextual_features import tokenize_and_preserve_labels
from choose_embed_type import choose_tokenizer_type
from load_pico import load_data, seed_everything

tqdm.pandas()


def build_features(seed):

    # load arguments
    args = getArguments() # get all the experimental arguments
    # seed 
    seed_everything( seed=seed )

    train_df, val_df, test_df = load_data(args.data_dir, seed=seed )
    print( 'Data loaded...' )

    start_feature_transformation = time.time()
    tokenizer, model = choose_tokenizer_type( args.embed )
    print('Tokenizer and model loaded...')

    # Get Contextual word features and POS features
    if 'mtl' not in args.model:
        train_toks, train_labs, train_pos, train_masks, train_claim_offsets = transform( train_df, tokenizer, args.max_len, args.embed, args )
        val_toks, val_labs, val_pos, val_masks, val_claim_offsets = transform( val_df, tokenizer, args.max_len, args.embed, args )
        test_toks, test_labs, test_pos, test_masks, test_claim_offsets = transform( test_df, tokenizer, args.max_len, args.embed, args )
    else:
        train_toks, train_labs, train_labs_fine, train_pos, train_masks, train_claim_offsets = transform_mtl( train_df, tokenizer, args.max_len, args.embed, args )
        val_toks, val_labs, val_labs_fine, val_pos, val_masks, val_claim_offsets = transform_mtl( val_df, tokenizer, args.max_len, args.embed, args )
        test_toks, test_labs, test_labs_fine, test_pos, test_masks, test_claim_offsets = transform_mtl( test_df, tokenizer, args.max_len, args.embed, args )     


    # Assign transformed features to OG dataframe
    train_df = train_df.assign(embeddings = pd.Series(train_toks).values, label_pads = pd.Series(train_labs).values, attn_masks = pd.Series(train_masks).values, inputpos = pd.Series(train_pos).values, inputoffs = pd.Series(train_claim_offsets).values)
    val_df = val_df.assign(embeddings = pd.Series(val_toks).values, label_pads = pd.Series(val_labs).values, attn_masks = pd.Series(val_masks).values, inputpos = pd.Series(val_pos).values, inputoffs = pd.Series(val_claim_offsets).values)
    test_df = test_df.assign(embeddings = pd.Series(test_toks).values, label_pads = pd.Series(test_labs).values, attn_masks = pd.Series(test_masks).values, inputpos = pd.Series(test_pos).values, inputoffs = pd.Series(test_claim_offsets).values)

    if 'mtl' in args.model:
        train_df = train_df.assign(label_fine_pads = pd.Series(train_labs_fine).values)
        val_df = val_df.assign(label_fine_pads = pd.Series(val_labs_fine).values)
        test_df = test_df.assign(label_fine_pads = pd.Series(test_labs_fine).values)        

    # Get Contextual char features and orthographic features
    print( 'One-hot encode the characters and fetch the orthographic encodings...' )
    train_df = transform_char( train_df )
    val_df = transform_char( val_df )
    test_df = transform_char( test_df )

    return train_df, val_df, test_df, tokenizer, model