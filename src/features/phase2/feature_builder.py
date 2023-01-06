# generic imports
import ast
import sys
import time
from build_contextual_features import transform

from tqdm.notebook import tqdm
import pandas as pd
from arguments import getArguments
from build_contextual_features import tokenize_and_preserve_labels
from choose_embed_type import choose_tokenizer_type
from load_pico import load_data, seed_everything

tqdm.pandas()


def build_features():

    # load arguments
    args = getArguments() # get all the experimental arguments

    # seed 
    seed_everything( int(args.seed) )

    train_df, val_df = load_data(args.data_dir)
    print( 'Data loaded...' )

    start_feature_transformation = time.time()
    tokenizer, model = choose_tokenizer_type( args.embed )
    print('Tokenizer and model loaded...')

    # Transform data
    train_toks, train_labs, train_pos, train_masks, train_claim_offsets = transform( train_df, tokenizer, args.max_len, args.embed, args )
    val_toks, val_labs, val_pos, val_masks, val_claim_offsets = transform( val_df, tokenizer, args.max_len, args.embed, args )

    # Assign transformed data to OG dataframe
    train_df = train_df.assign(embeddings = pd.Series(train_toks).values, label_pads = pd.Series(train_labs).values, attn_masks = pd.Series(train_masks).values, inputpos = pd.Series(train_pos).values, inputoffs = pd.Series(train_claim_offsets).values)
    val_df = val_df.assign(embeddings = pd.Series(val_toks).values, label_pads = pd.Series(val_labs).values, attn_masks = pd.Series(val_masks).values, inputpos = pd.Series(val_pos).values, inputoffs = pd.Series(val_claim_offsets).values)

    return train_df, val_df, tokenizer, model