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


class MTL_4(nn.Module):

    def __init__(self, freeze_bert, tokenizer, model, exp_args):
        super(MTL_4, self).__init__()

        self.tokenizer = tokenizer

        # Define the dimentions for the word features 
        self.batch_size = exp_args.batch # 10
        self.max_len = exp_args.max_len # 512 maximum sequence length
        transformer_dim = 768 # output by transformers
        self.hidden_transformer = transformer_dim * 2

        pos_dim = 18
        self.hidden_pos = pos_dim * 2 # If bilstm processed POS, then hidden_pos = (hidden_pos * 2)

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

        self.n_conv_filters=256
        self.conv_out_dim = 1024

        # char layers
        self.conv1 = nn.Conv1d( char_hidden_dim, self.n_conv_filters, kernel_size=7, padding=0)
        # Max Pooling Layer
        self.max_pooling_layer = nn.MaxPool1d( 24 )
        # fc layer
        self.fc1 = nn.Sequential( nn.Linear( self.n_conv_filters, self.conv_out_dim ) , nn.Dropout(0.5) )

        # log reg (for coarse labels)
        self.hidden2tag = nn.Linear( transformer_dim + pos_dim + self.conv_out_dim , exp_args.num_labels)
        # log reg (for fine labels)
        self.hidden2tag_fine = nn.Linear(transformer_dim + pos_dim + self.conv_out_dim, exp_args.num_labels * 2)

        # loss calculation (for coarse labels)
        self.loss_fct = nn.CrossEntropyLoss()
        # loss calculation (for coarse labels)
        self.loss_fct_fine = nn.CrossEntropyLoss()


    def forward(self, input_ids=None, attention_mask=None, labels=None, labels_fine=None, input_pos=None, input_offs=None, input_char_encode=None, input_char_ortho=None, mode = None, args = None):

        self.input_pos = input_pos

        # Transformer
        outputs = self.transformer_layer(
            input_ids,
            attention_mask = attention_mask
        )
        # output 0 = batch size 6, tokens MAX_LEN, each token dimension 768 [CLS] token
        sequence_output = outputs[0]

        # combine both transformer output and POS tags (onehot encoded)
        transform_pos_output = torch.cat( (sequence_output, self.input_pos), 2)
        transform_pos_output = transform_pos_output.cuda()


        # mask the unimportant tokens before log_reg (NOTE: CLS token (position 0) is not masked!!!)
        mask = (
            (input_ids[:, :transform_pos_output.shape[1]] != self.tokenizer.pad_token_id)
            & (input_ids[:, :transform_pos_output.shape[1]] != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )

        # mask the transformer output and labels - coarse and fine
        mask_expanded = mask.unsqueeze(-1).expand(transform_pos_output.size())
        transform_pos_output *= mask_expanded.float()
        labels *= mask.long()
        labels_fine *= mask.long()

        # character CNN
        # conv layer
        input_char_encode_reshaped = input_char_encode.reshape( input_char_encode.shape[0], input_char_encode.shape[1], input_char_encode.shape[3], input_char_encode.shape[2] )
        input_char_encode_reshaped = input_char_encode_reshaped.cuda()

        # empty tensor to collect the convolutions
        batch_conv = torch.empty(size= ( 0, self.max_len, 1024 ))
        batch_conv = batch_conv.cuda()

        # iterate through each sentence to retrieve the character representation
        for i in range( input_char_ortho.shape[0] ):

            # print( char_concat_output[i].shape )

            conv1_output = self.conv1( input_char_encode_reshaped[i].float()  )
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
        transform_pos_char_output = torch.cat( ( transform_pos_output, batch_conv), 2)
        transform_pos_char_output = transform_pos_char_output.cuda()

        # linear for coarse
        probablity = F.relu ( self.hidden2tag( transform_pos_char_output ) )
        max_probs = torch.max(probablity, dim=2)         
        logits = max_probs.indices

        # linear for fine
        probablity_fine = F.relu ( self.hidden2tag_fine( transform_pos_char_output ) )
        max_probs_fine = torch.max(probablity_fine, dim=2)
        # logits_fine = max_probs_fine.indices.flatten()
        logits_fine = max_probs_fine.indices

        # calculate coarse loss
        loss_coarse = self.loss_fct(probablity.view(-1, args.num_labels), labels.view(-1))
        # print('Coarse-grained loss: ', loss_coarse)

        # calculate fine loss
        loss_fine = self.loss_fct_fine(probablity_fine.view(-1, args.num_labels * 2), labels_fine.view(-1))
        # print('Fine-grained loss: ', loss_fine)

        loss = loss_coarse + loss_fine

        return loss_coarse, logits, labels, loss_fine, logits_fine, labels_fine, mask, loss