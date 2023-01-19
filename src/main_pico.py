__author__ = "Dhrangadhariya, Anjani"
__credits__ = ["Dhrangadhariya, Anjani"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Dhrangadhariya, Anjani"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__version__ = "1.0"

from ast import arg
import os
import random
import sys
import time
import traceback

path = '/home/anjani/pico-semeval2023/src/features/phase2'
sys.path.append(path)
print(sys.path)
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
from torchcontrib.optim import SWA
from torch.optim import AdamW


# Transformers 
from transformers import (AdamW, AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer, RobertaConfig,
                          RobertaModel, get_linear_schedule_with_warmup,
                          logging)

from features.phase2 import arguments, choose_embed_type, feature_builder

logging.set_verbosity_error()

from train_pico import evaluate, train
from train_pico_mtl import train_mtl

# Set up the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# mlflow 
from utilities.mlflow_logging import *
from load_pico import seed_everything


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
        for i in [0, 1, 42]:

            # get arguments
            exp_args = arguments.getArguments() # get all the experimental arguments
            seed_everything( i )

            # This is executed after the seed is set because it is imperative to have reproducible data run after shuffle
            train_df, val_df, test_df, tokenizer, model = feature_builder.build_features( i )
            print( 'Train and validation dataframes loaded...' )

            # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
            train_input_ids = convertDf2Tensor(train_df['embeddings'], np.int64)
            train_input_labels = convertDf2Tensor(train_df['label_pads'], np.int64)
            if 'mtl' in exp_args.model: 
                train_input_labels_fine = convertDf2Tensor(train_df['label_fine_pads'], np.int64)
            train_attn_masks = convertDf2Tensor(train_df['attn_masks'], np.int64)
            train_pos_tags = convertDf2Tensor(train_df['inputpos'], np.int64)
            train_offsets = convertDf2Tensor(train_df['inputoffs'], np.int64)
            train_char  = convertDf2Tensor(train_df['char_encode'], np.int64)
            train_ortho  = convertDf2Tensor(train_df['char_ortho'], np.int64)

            dev_input_ids = convertDf2Tensor( val_df['embeddings'], np.int64)
            dev_input_labels = convertDf2Tensor( val_df['label_pads'], np.int64)
            if 'mtl' in exp_args.model: 
                dev_input_labels_fine = convertDf2Tensor(val_df['label_fine_pads'], np.int64)
            dev_attn_masks = convertDf2Tensor( val_df['attn_masks'], np.int64)
            dev_pos_tags = convertDf2Tensor( val_df['inputpos'], np.int64)
            dev_offsets = convertDf2Tensor(val_df['inputoffs'], np.int64)
            dev_char  = convertDf2Tensor(val_df['char_encode'], np.int64)
            dev_ortho  = convertDf2Tensor(val_df['char_ortho'], np.int64)

            test_input_ids = convertDf2Tensor( test_df['embeddings'], np.int64)
            test_input_labels = convertDf2Tensor( test_df['label_pads'], np.int64)
            if 'mtl' in exp_args.model: 
                test_input_labels_fine = convertDf2Tensor(test_df['label_fine_pads'], np.int64)
            test_attn_masks = convertDf2Tensor( test_df['attn_masks'], np.int64)
            test_pos_tags = convertDf2Tensor( test_df['inputpos'], np.int64)
            test_offsets = convertDf2Tensor(test_df['inputoffs'], np.int64)
            test_char  = convertDf2Tensor(test_df['char_encode'], np.int64)
            test_ortho  = convertDf2Tensor(test_df['char_ortho'], np.int64)
            print( 'Tensors loaded...' )

            # # ##################################################################################
            # # One-hot encode POS tags
            # # ##################################################################################
            train_pos_tags = torch.nn.functional.one_hot(train_pos_tags, num_classes= - 1)
            dev_pos_tags = torch.nn.functional.one_hot(dev_pos_tags, num_classes= - 1)
            test_pos_tags = torch.nn.functional.one_hot(test_pos_tags, num_classes= - 1)

            # # ##################################################################################
            # # Create dataloaders from the tensors
            # # ##################################################################################
            # # Create the DataLoader for our training, validation and test set.
            if 'mtl' not in exp_args.model:
                train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks, train_pos_tags, train_offsets, train_char, train_ortho)
                dev_data = TensorDataset(dev_input_ids, dev_input_labels, dev_attn_masks, dev_pos_tags, dev_offsets, dev_char, dev_ortho)
                test_data = TensorDataset(test_input_ids, test_input_labels, test_attn_masks, test_pos_tags, test_offsets, test_char, test_ortho)
            elif 'mtl' in exp_args.model:
                train_data = TensorDataset(train_input_ids, train_input_labels, train_input_labels_fine, train_attn_masks, train_pos_tags, train_offsets, train_char, train_ortho)
                dev_data = TensorDataset(dev_input_ids, dev_input_labels, dev_input_labels_fine, dev_attn_masks, dev_pos_tags, dev_offsets, dev_char, dev_ortho)
                test_data = TensorDataset(test_input_ids, test_input_labels, test_input_labels_fine, test_attn_masks, test_pos_tags, test_offsets, test_char, test_ortho)

            train_dataloader = DataLoader(train_data, sampler=None, batch_size=10, shuffle=False)
            dev_dataloader = DataLoader(dev_data, sampler=None, batch_size=exp_args.batch, shuffle=False)
            test_dataloader = DataLoader(test_data, sampler=None, batch_size=exp_args.batch, shuffle=False)
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

            # base_opt = torch.optim.AdamW(model.parameters(), lr=0.1)
            # optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)

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

            print('##################################################################################')
            print('Begin training...')
            print('##################################################################################')
            train_start = time.time()
            if 'mtl' not in exp_args.model:
                saved_models = train(loaded_model, tokenizer, optimizer, scheduler, train_dataloader, dev_dataloader, exp_args, seed = i)
            elif 'mtl' in exp_args.model:
                saved_models = train_mtl(loaded_model, tokenizer, optimizer, scheduler, train_dataloader, dev_dataloader, exp_args)

            print("--- Took %s seconds to train and evaluate the model ---" % (time.time() - train_start))

            print('##################################################################################')
            print('Begin test...')
            print('##################################################################################')
            # Print the experiment details
            print('The experiment on ', exp_args.entity, ' entity class using ', exp_args.embed, ' running for ', exp_args.max_eps, ' epochs.'  )
            print( 'Results for the seed: ', i )
            print( 'Arguments: ', exp_args )

            checkpoint = torch.load(saved_models[-1], map_location='cuda:0')
            # checkpoint = torch.load('/mnt/nas2/results/Results/systematicReview/SemEval2023/models/all/roberta_epoch_0.pth', map_location='cuda:0')
            model.load_state_dict( checkpoint, strict=False  )
            model = torch.nn.DataParallel(model, device_ids=[0])

            # print('Applying the best model on test set ...')
            # test1_cr, all_pred_flat, all_GT_flat, cm1, test1_words, class_rep_temp = evaluate(model, tokenizer, optimizer, scheduler, test_dataloader, exp_args)
            # print(test1_cr)
            # print(cm1)


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
