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
from features.phase2 import feature_builder, choose_embed_type, arguments

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

from train_pico import train

# Set up the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# mlflow 
from utilities.mlflow_logging import *

def loadModel(model, exp_args):

    if exp_args.parallel == 'true':

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids = [0, 1])
            print("Using", str(model.device_ids), " GPUs!")
            return model

    elif exp_args.parallel == 'false':
        model = nn.DataParallel(model, device_ids = [0])
        return model

def convertDf2Tensor(df, data_type):

    if data_type==np.float64:
        # return torch.from_numpy( np.array( list( [ [0.0, 0.0], [0.0, 0.0] ] ) , dtype=data_type ) ).clone().detach()
        return torch.from_numpy( np.array( list( df ), dtype=float ) ).clone().detach()

    else:
        return torch.from_numpy( np.array( list( df ), dtype=data_type ) ).clone().detach()


if __name__ == "__main__":

    try:

        # with mlflow.start_run() as run:
        for i in ['11']:

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
            train_offsets = convertDf2Tensor(train_df['inputoffs'], np.int64)

            dev_input_ids = convertDf2Tensor( val_df['embeddings'], np.int64)
            if exp_args.supervision == 'ws': 
                dev_input_labels = convertDf2Tensor( val_df['label_pads'], np.float64)
            elif exp_args.supervision == 'fs': 
                dev_input_labels = convertDf2Tensor( val_df['label_pads'], np.int64)
            dev_attn_masks = convertDf2Tensor( val_df['attn_masks'], np.int64)
            dev_pos_tags = convertDf2Tensor( val_df['inputpos'], np.int64)
            dev_offsets = convertDf2Tensor(val_df['inputoffs'], np.int64)
            print( 'Tensors loaded...' )

            # # ----------------------------------------------------------------------------------------
            # # One-hot encode POS tags
            # # ----------------------------------------------------------------------------------------
            train_pos_tags = torch.nn.functional.one_hot(train_pos_tags, num_classes= - 1)
            dev_pos_tags = torch.nn.functional.one_hot(dev_pos_tags, num_classes= - 1)

            # # ----------------------------------------------------------------------------------------
            # # Create dataloaders from the tensors
            # # ----------------------------------------------------------------------------------------
            # # Create the DataLoader for our training set.
            train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks, train_pos_tags, train_offsets)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=None, batch_size=10, shuffle=False)

            # # Create the DataLoader for our validation set.
            dev_data = TensorDataset(dev_input_ids, dev_input_labels, dev_attn_masks, dev_pos_tags, dev_offsets)
            dev_sampler = RandomSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=None, batch_size=10, shuffle=False)
            print( 'Dataloaders loaded...' )

            # ##################################################################################
            # #Instantiating the BERT model
            # ##################################################################################
            print("Building model...")
            createOSL = time.time()
            loaded_model = choose_embed_type.choose_model(exp_args.embed, tokenizer, model, exp_args.model, exp_args)

            # ##################################################################################
            # # Tell pytorch to run data on this model on the GPU and parallelize it
            # ##################################################################################

            if exp_args.train_from_scratch == True:
                loaded_model = loadModel(loaded_model, exp_args)
            else:
                checkpoint = torch.load(exp_args.plugin_model, map_location='cuda:0')
                loaded_model.load_state_dict( checkpoint )
                loaded_model = loadModel(loaded_model, exp_args)
            print('The devices used: ', str(loaded_model.device_ids) )


            # ##################################################################################
            # # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
            optimizer = AdamW(model.parameters(),
                            lr = exp_args.lr, # args.learning_rate - default is 5e-5 (for BERT-base)
                            eps = exp_args.eps, # args.adam_epsilon  - default is 1e-8.
                            )

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(train_dataloader) * exp_args.max_eps
            print('Total steps per epoch: ', total_steps)

            # Create the learning rate scheduler.
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=total_steps*exp_args.lr_warmup,
                                                        num_training_steps = total_steps)

            # print("Created the optimizer, scheduler and loss function objects in {} seconds".format(time.time() - st))
            print("--- Took %s seconds to create the model, optimizer, scheduler and loss function objects ---" % (time.time() - createOSL))

            # Log the parameters
            if exp_args.log == True:
                logParams(exp_args)

            # print('##################################################################################')
            # print('Begin training...')
            # print('##################################################################################')
            train_start = time.time()
            saved_models = train(loaded_model, tokenizer, optimizer, scheduler, train_dataloader, dev_dataloader, exp_args)
            print("--- Took %s seconds to train and evaluate the model ---" % (time.time() - train_start))


    except Exception as ex:

        template = "An exception of type {0} occurred. Arguments:{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print( message )

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())

        # logging.info(message)
        string2log = str(exc_type) + ' : ' + str(fname) + ' : ' + str(exc_tb.tb_lineno)
        # logging.info(string2log)
        # logging.info(traceback.format_exc())