# generic imports
import ast
import sys
import time
from build_contextual_features import transform

from tqdm.notebook import tqdm

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

    # tokenize and preserve labels
    transform( train_df, tokenizer, args.max_len, args.embed )

    return train_df, val_df


build_features()
