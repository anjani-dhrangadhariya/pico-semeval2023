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
from torch.nn import Dropout

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

class ENSEMBLE5(nn.Module):

    def __init__(self, freeze_bert, tokenizer, model, exp_args):
        super(ENSEMBLE5, self).__init__()

        self.tokenizer = tokenizer

        #Instantiating BERT model object 
        self.transformer_layer = model
        transformer_dim = 768
        pos_dim = 18
        hidden_pos = pos_dim * 2

        # Define the dimentions for the character features 
        char_hidden_dim = 69
        ortho_hidden_dim = 4
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.transformer_layer.parameters():
                p.requires_grad = False

        # dropout to regularize
        # self.dropout_layer = Dropout(p=0.8, inplace=False)

        # POS encoder
        if exp_args.bidrec == True and exp_args.pos_encoding == 'lstm':
            self.lstmpos_layer = nn.LSTM(input_size=pos_dim, hidden_size = hidden_pos, num_layers = 1, bidirectional=exp_args.bidrec, batch_first=True)

        # Character layers
        # Concatenate the ortho and char encoding layers
        char_dim = char_hidden_dim + ortho_hidden_dim # 69 + 4 = 73
        self.n_conv_filters=256
        self.conv_out_dim = 1024

        # char layers (ensemble5 only takes ortho features)
        self.conv1 = nn.Conv1d( ortho_hidden_dim, self.n_conv_filters, kernel_size=7, padding=0)
        # Max Pooling Layer
        self.max_pooling_layer = nn.MaxPool1d( 24 )
        # fc layer
        self.fc1 = nn.Sequential( nn.Linear( self.n_conv_filters, self.conv_out_dim ) , nn.Dropout(0.5) )


        # log reg predictor
        if exp_args.bidrec == True and exp_args.pos_encoding == 'lstm':
            self.hidden2tag = nn.Linear(transformer_dim + ( hidden_pos * 2 ) + self.conv_out_dim, exp_args.num_labels)
        else:
            self.hidden2tag = nn.Linear( transformer_dim + pos_dim + self.conv_out_dim , exp_args.num_labels)

        if exp_args.predictor == 'crf':
            # crf layer
            self.crf_layer = CRF(exp_args.num_labels, batch_first=True)

        # loss calculation
        self.loss_fct = nn.CrossEntropyLoss()

    def get_loss(self, probablities_masked, labels_masked ):

        cumulative_loss = torch.cuda.FloatTensor([0.0]) 
        for i in range(0, probablities_masked.shape[0]):
            loss = self.loss_fct( probablities_masked[i] , labels_masked[i]  )
            cumulative_loss += loss
        
        average_loss = cumulative_loss /  probablities_masked.shape[0]

        return average_loss

    # def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None, input_offs=None, mode = None, args = None):
    def forward(self, input_ids=None, attention_mask=None, labels=None, labels_fine=None, input_pos=None, input_offs=None, input_char_encode=None, input_char_ortho=None, mode = None, args = None):

        self.max_len = args.max_len

        # Transformer
        outputs = self.transformer_layer(
            input_ids,
            attention_mask = attention_mask
        )
        sequence_output = outputs[0]

        # dropout layer
        # sequence_output = self.dropout_layer( sequence_output )

        # LSTM encode the POS tags
        if args.pos_encoding == 'lstm':
            packed_pos_input, pos_perm_idx, pos_seq_lengths, total_length_pos = get_packed_padded_output_dataparallel(input_pos.float(), attention_mask)
            self.lstmpos_layer.flatten_parameters()
            packed_pos_output, (ht_pos, ct_pos) = self.lstmpos_layer(packed_pos_input)
            pos_output, pos_input_sizes = pad_packed_sequence(packed_pos_output, batch_first=True, total_length=total_length_pos)
            _, unperm_idx = pos_perm_idx.sort(0)
            lstm_pos_output = pos_output[unperm_idx]

        # Concat POS tags with embedding output
        if args.pos_encoding == 'lstm':
            sequence_output_concat = torch.cat( (sequence_output, lstm_pos_output), 2) # concatenate at dim 2 for embeddings and tags
            sequence_output_concat = sequence_output_concat.cuda()
        if args.pos_encoding == 'onehot':
            sequence_output_concat = torch.cat( (sequence_output, input_pos), 2) # concatenate at dim 2 for embeddings and tags
            sequence_output_concat = sequence_output_concat.cuda()
        
        # mask the unimportant tokens before log_reg
        mask = (
            (input_ids != self.tokenizer.pad_token_id)
            & (input_ids != self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token))
            & (input_ids != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )

        # expand the mask according the concatenated sequence output and POS tags
        mask_expanded = mask.unsqueeze(-1).expand(sequence_output_concat.size())
        sequence_output_concat *= mask_expanded.float()
        labels_masked = labels * mask.long()

        # conv layer
        input_char_ortho_reshaped = input_char_ortho.reshape( input_char_ortho.shape[0], input_char_ortho.shape[1], input_char_ortho.shape[3], input_char_ortho.shape[2] )
        input_char_ortho_reshaped = input_char_ortho_reshaped.cuda()

        # empty tensor to collect the convolutions
        batch_conv = torch.empty(size= ( 0, self.max_len, 1024 ))
        batch_conv = batch_conv.cuda()

        # iterate through each sentence to retrieve the character representation
        for i in range( input_char_ortho.shape[0] ):

            # print( char_concat_output[i].shape )

            conv1_output = self.conv1( input_char_ortho_reshaped[i].float()  )
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


        # concat conv output with transformer+POS embeddings
        transform_pos_char_output = torch.cat( ( sequence_output_concat, batch_conv), 2)
        transform_pos_char_output = transform_pos_char_output.cuda()

        # linear layer (log reg) to emit class probablities
        probablities = F.relu ( self.hidden2tag( transform_pos_char_output ) )
        probablities_mask_expanded = mask.unsqueeze(-1).expand(probablities.size())
        probablities_masked = probablities * probablities_mask_expanded.float()

        if args.predictor == 'linear':

            # cumulative_loss = torch.cuda.FloatTensor([0.0]) 
            # for i in range(0, probablities.shape[0]):
            #     loss = self.loss_fct( probablities_masked[i] , labels_masked[i]  )
            #     cumulative_loss += loss
            
            # average_loss = cumulative_loss /  probablities.shape[0]

            average_loss = self.get_loss( probablities_masked, labels_masked )

            return average_loss, probablities, probablities_mask_expanded, labels, mask, mask

        else:

            # on the first time steps XXX CLS token is active at position 0
            # for eachIndex in range( mask.shape[0] ):
            #     mask[eachIndex, 0] = True

            # for eachIndex in range( labels_masked.shape[0] ):
            #     labels_masked[eachIndex, 0] = 0
            #     labels[eachIndex, 0] = 0

            # CRF emissions
            loss = self.crf_layer(probablities_masked, labels_masked, reduction='token_mean', mask = None)

            emissions_ = self.crf_layer.decode( probablities_masked , mask = None)
            emissions = [item for sublist in emissions_ for item in sublist] # flatten the nest list of emissions

            target_emissions = torch.zeros(probablities_masked.shape[0], probablities_masked.shape[1])
            target_emissions = target_emissions.cuda()
            for eachIndex in range( target_emissions.shape[0] ):
                target_emissions[ eachIndex, :torch.tensor( emissions_[eachIndex] ).shape[0] ] = torch.tensor( emissions_[eachIndex] )
            
            return loss, target_emissions, probablities_mask_expanded, labels, mask, mask