__author__ = "Dhrangadhariya, Anjani"
__credits__ = ["Dhrangadhariya, Anjani"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Dhrangadhariya, Anjani"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__version__ = "1.0"

import os
import random
import sys
import time
import traceback


path = '/home/anjani/pico-semeval2023/src/features/phase2'
sys.path.append(path)
print(sys.path)
from features.phase2 import feature_builder
from features.phase2 import arguments

import mlflow
import numpy as np
import pandas as pd

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_artifacts, log_metric, log_param
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaConfig, RobertaModel
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

def convertDf2Tensor(df, data_type):

    if data_type==np.float64:

        # return torch.from_numpy( np.array( list( [ [0.0, 0.0], [0.0, 0.0] ] ) , dtype=data_type ) ).clone().detach()
        return torch.from_numpy( np.array( list( df ), dtype=float ) ).clone().detach()

    else:
        return torch.from_numpy( np.array( list( df ), dtype=data_type ) ).clone().detach()


if __name__ == "__main__":

    try:

        # get arguments
        exp_args = arguments.getArguments() # get all the experimental arguments

        # This is executed after the seed is set because it is imperative to have reproducible data run after shuffle
        train_df, val_df, tokenizer, model = feature_builder.build_features()
        print( 'Train and validation dataframes loaded...' )

        # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
        train_input_ids = convertDf2Tensor(train_df['embeddings'], np.int64)
        if exp_args.supervision == 'ws': 
            train_input_labels = convertDf2Tensor(train_df['label_pads'], np.float64)
        elif exp_args.supervision == 'fs': 
            train_input_labels = convertDf2Tensor(train_df['label_pads'], np.int64)
        train_attn_masks = convertDf2Tensor(train_df['attn_masks'], np.int64)
        train_pos_tags = convertDf2Tensor(train_df['inputpos'], np.int64)

        dev_input_ids = convertDf2Tensor( val_df['embeddings'], np.int64)
        if exp_args.supervision == 'ws': 
            dev_input_labels = convertDf2Tensor( val_df['label_pads'], np.float64)
        elif exp_args.supervision == 'fs': 
            dev_input_labels = convertDf2Tensor( val_df['label_pads'], np.int64)
        dev_attn_masks = convertDf2Tensor( val_df['attn_masks'], np.int64)
        dev_pos_tags = convertDf2Tensor( val_df['attn_masks'], np.int64)
        print( 'Tensors loaded...' )

        



    except Exception as ex:

        template = "An exception of type {0} occurred. Arguments:{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print( message )

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())

        logging.info(message)
        string2log = str(exc_type) + ' : ' + str(fname) + ' : ' + str(exc_tb.tb_lineno)
        logging.info(string2log)
        logging.info(traceback.format_exc())