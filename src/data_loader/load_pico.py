# generic imports
import pandas as pd

from arguments import getArguments

# get arguments
args = getArguments() # get all the experimental arguments

# Load dataframe with PICO
def load_data(input_directory):

    train = 'st2_train_preprocessed.tsv'
    val = 'st2_val_preprocessed.tsv'

    train_df = pd.read_csv(f'{input_directory}/{train}', sep='\t')
    val_df = pd.read_csv(f'{input_directory}/{val}', sep='\t')

    return train, val

train_df, val_df = load_data(args.data_dir)
print( 'Data loaded...' )


# Get features

