'''
Model with BERT as embedding layer followed by a CRF decoder
'''
__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__status__ = "Prototype/Research"

##################################################################################
# Imports
##################################################################################
# staple imports
from cProfile import label
from multiprocessing import reduction
import warnings

import sys

from train_pico import constrained_beam_search
path = '/home/anjani/pico-semeval2023/src/models/phase2'
sys.path.append(path)
path = '/home/anjani/pico-semeval2023/src/utilities/'
sys.path.append(path)
from models.loss import cross_entropy_with_probs
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import glob
import numpy as np
import pandas as pd
import time
import datetime
import argparse
import pdb
import json
import random
import statistics

# numpy essentials
from numpy import asarray
import numpy as np

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# keras essentials
from keras.preprocessing.sequence import pad_sequences

# sklearn
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix

# pyTorch CRF
from torchcrf import CRF

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW, BertConfig 
from transformers import get_linear_schedule_with_warmup

# Import data getters
from utilities.helper_functions import get_packed_padded_output, get_packed_padded_output_dataparallel


class MTL_0(nn.Module):

    def __init__(self, freeze_bert, tokenizer, model, exp_args):
        super(MTL_0, self).__init__()

        self.tokenizer = tokenizer

        # Define the dimentions for the word features 
        self.batch_size = exp_args.batch # 10
        self.max_len = exp_args.max_len # 512 maximum sequence length
        transformer_dim = 768 # output by transformers
        self.hidden_transformer = transformer_dim

        pos_dim = 18
        self.hidden_pos = pos_dim # If bilstm processed POS, then hidden_pos = (hidden_pos * 2)

        # Define the dimentions for the character features 
        char_hidden_dim = 69
        ortho_hidden_dim = 4

        ############################################################################
        # Define the layers
        ############################################################################
        # Word layers
        #Instantiating BERT model object 
        self.transformer_layer = model
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.transformer_layer.parameters():
                p.requires_grad = False

        # Dims pos - 512,18
        # Dims transform = 512, 768
        # BiLSTM encode the concatenated word features 
        self.lstm_layer = nn.LSTM(input_size = (transformer_dim + pos_dim), hidden_size = ( self.hidden_transformer + self.hidden_pos ) , num_layers = 1, bidirectional=exp_args.bidrec, batch_first=True)
        # Output of LSTM layer will be the word representation which will be merged with the character embeddings

        # Character layers
        # Concatenate the ortho and char encoding layers
        char_dim = char_hidden_dim + ortho_hidden_dim # 69 + 4 = 73
        self.n_conv_filters=256

        # conv layer
        self.conv1 = nn.Conv1d( char_dim, self.n_conv_filters, kernel_size=7, padding=0)
        # Max Pooling Layer
        self.max_pooling_layer = nn.MaxPool1d( 24 )
        # fc layer
        self.fc1 = nn.Sequential( nn.Linear( self.n_conv_filters, 1024 ) , nn.Dropout(0.5) )

        ################################################################
        self.hidden2tag_coarse = nn.Linear(2596, exp_args.num_labels)
        self.hidden2tag_coarse = nn.Linear(2596, exp_args.num_labels)

        # loss calculation
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None, input_offs=None, input_char_encode=None, input_char_ortho=None, mode = None, args = None):

        self.input_pos = input_pos

        # Transformer
        outputs = self.transformer_layer(
            input_ids,
            attention_mask = attention_mask
        )
        # output 0 = batch size 6, tokens MAX_LEN, each token dimension 768 [CLS] token
        sequence_output = outputs[0]

        # print( 'Dimensions before concatenation (transformer): ', sequence_output.shape )
        # print( 'Dimensions before concatenation (POS): ', self.input_pos.shape )
        # print( 'Dimention of attention mask: ', attention_mask.shape )

        transform_pos_output = torch.cat( (sequence_output, self.input_pos), 2)
        transform_pos_output = transform_pos_output.cuda()

        # print( 'Dimensions after concatenation: ', transform_pos_output.shape )

        # use BiLSTM layer on the concatenated word features
        packed_input, perm_idx, seq_lengths = get_packed_padded_output(transform_pos_output, attention_mask, self.tokenizer)
        packed_output, (ht, ct) = self.lstm_layer( packed_input )

        # Unpack and reorder the output
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = output[unperm_idx] # lstm_output.shape = shorter than the padded torch.Size([6, 388, 512])
        seq_lengths_ordered = seq_lengths[unperm_idx]
        # print( 'Shape of LSTM output: ', lstm_output.shape )
        
        # expand the shortened lstm output
        lstm_repadded = torch.zeros(size= ( lstm_output.shape[0], self.max_len, ( (self.hidden_transformer *2) + (self.hidden_pos*2) ) ))
        lstm_repadded[ :, :lstm_output.shape[1], :lstm_output.shape[2] ] = lstm_output
        lstm_repadded = lstm_repadded.cuda()


        # mask the unimportant tokens before log_reg (NOTE: CLS token (position 0) is not masked!!!)
        mask = (
            (input_ids[:, :lstm_repadded.shape[1]] != self.tokenizer.pad_token_id)
            & (input_ids[:, :lstm_repadded.shape[1]] != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )

        # on the first time steps XXX CLS token is active at position 0
        for eachIndex in range( mask.shape[0] ):
            mask[eachIndex, 0] = True

        for eachIndex in range( labels.shape[0] ):
            labels[eachIndex, 0] = 0
        
        # mask the lstm and labels
        mask_expanded = mask.unsqueeze(-1).expand(lstm_repadded.size())
        lstm_repadded *= mask_expanded.float()
        labels *= mask.long()

        ###############################################################################################

        # Concatenate character features
        # print('Char encoding shape: ' , input_char_encode.shape) 
        # print('Ortho encoding shape: ' , input_char_ortho.shape) 

        char_concat_output = torch.cat( (input_char_encode, input_char_ortho), 3)
        char_concat_output = char_concat_output.cuda()
        # print( 'Dimension of the concatenated char output: ', char_concat_output.shape )

        # conv layer
        char_concat_output = char_concat_output.reshape( char_concat_output.shape[0], char_concat_output.shape[1], char_concat_output.shape[3], char_concat_output.shape[2] )
        char_concat_output = char_concat_output.cuda()

        # empty tensor to collect the convolutions
        batch_conv = torch.empty(size= ( 0, self.max_len, 1024 ))
        batch_conv = batch_conv.cuda()

        # iterate through each sentence to retrieve the character representation
        for i in range( char_concat_output.shape[0] ):

            # print( char_concat_output[i].shape )

            conv1_output = self.conv1( char_concat_output[i].float()  )
            # print( 'Dimension of the conv1 output: ', conv1_output.shape )

            # GAP
            max_pool_out = self.max_pooling_layer(conv1_output)
            # print( 'Dimension of the max_pool_out output: ', max_pool_out.shape )
                        
            max_pool_out = max_pool_out.squeeze()
            # print( 'Dimension of the sentence embeddings: ', max_pool_out.shape )

            # fully connected
            fc1_output = F.relu ( self.fc1( max_pool_out ) )
            # print( 'Dimension of the fully connected output: ', fc1_output.shape )

            batch_conv = torch.cat( ( batch_conv, fc1_output.unsqueeze(dim=0) ) )

        #######
        # print( 'Shape of the convoluted batch: ', batch_conv.shape )

        # combine both LSTM and CNN outputs
        word_char_features = torch.cat( (batch_conv, lstm_repadded), 2)
        # print( 'After combining character and word features: ', word_char_features.shape )


        # predict on top of the combined word+char them
        probablities = F.relu ( self.hidden2tag( word_char_features ) )

        cumulative_loss = torch.cuda.FloatTensor([0.0]) 

        for i in range(0, probablities.shape[0]):

            loss = self.loss_fct( probablities[i] , labels[i]  )

            cumulative_loss += loss
        
        average_loss = cumulative_loss /  probablities.shape[0]

        return average_loss, probablities, mask_expanded, labels, mask, mask