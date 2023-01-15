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

class TRANSFORMERPOSCRF(nn.Module):

    def __init__(self, freeze_bert, tokenizer, model, exp_args):
        super(TRANSFORMERPOSCRF, self).__init__()
        #Instantiating BERT model object 
        self.transformer_layer = model

        transformer_dim = 768
        pos_dim = 18
        self.hidden_pos = 36
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.transformer_layer.parameters():
                p.requires_grad = False

        self.tokenizer = tokenizer

        # POS encoder
        if exp_args.bidrec == True and exp_args.pos_encoding == 'lstm':
            self.lstmpos_layer = nn.LSTM(input_size=pos_dim, hidden_size = self.hidden_pos, num_layers = 1, bidirectional=exp_args.bidrec, batch_first=True)

        # log reg
        if exp_args.pos_encoding == 'lstm':
            self.hidden2tag = nn.Linear(transformer_dim + (self.hidden_pos* 2), exp_args.num_labels)
        else:
            self.hidden2tag = nn.Linear(transformer_dim + pos_dim, exp_args.num_labels)

        # crf
        self.crf_layer = CRF(exp_args.num_labels, batch_first=True)
        
        # loss calculation
        self.loss_fct = nn.CrossEntropyLoss()

    
    def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None, input_offs=None, mode = None, args = None):

        # initialize variables
        input_offs = input_offs

        # Transformer
        outputs = self.transformer_layer(
            input_ids,
            attention_mask = attention_mask
        )
        # print('Shape of the transformer output: ', len(outputs))

        # output 0 = batch size 6, tokens MAX_LEN, each token dimension 768 [CLS] token
        sequence_output = outputs[0]

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
            sequence_output = torch.cat( (sequence_output, lstm_pos_output), 2) # concatenate at dim 2 for embeddings and tags
            sequence_output = sequence_output.cuda()
        if args.pos_encoding == 'onehot':
            sequence_output = torch.cat( (sequence_output, input_pos), 2) # concatenate at dim 2 for embeddings and tags
            sequence_output = sequence_output.cuda()

        # mask the unimportant tokens before CRF
        mask = (
            (input_ids != self.tokenizer.pad_token_id)
            & (input_ids != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )

        # print( 'Mask before expansion: ', mask.shape )
        mask_expanded = mask.unsqueeze(-1).expand(sequence_output.size())
        # print( 'Mask after expansion: ', mask_expanded.shape )

        sequence_output *= mask_expanded.float()
        # print( 'Masked transformer output: ', sequence_output.shape )

        if mask.shape == labels.shape:
            labels_masked = labels * mask.long()
            offsets_masked = input_offs * mask.long()
        else:
            label_masks_expanded = mask.unsqueeze(-1).expand(labels.size())
            labels_masked = labels * label_masks_expanded.long()
            offsets_masked = input_offs * label_masks_expanded.long()

        # linear layer (log reg) to emit class probablities
        probablities = F.relu ( self.hidden2tag( sequence_output ) )
        # print( 'probablities: ', probablities.shape )
        probablities_mask_expanded = mask.unsqueeze(-1).expand(probablities.size())
        probablities_masked = probablities * probablities_mask_expanded.float()
        # print( 'probablities masked: ', probablities_masked.shape )

        # on the first time steps XXX CLS token is active at position 0
        for eachIndex in range( mask.shape[0] ):
            mask[eachIndex, 0] = True

        for eachIndex in range( labels_masked.shape[0] ):
            labels_masked[eachIndex, 0] = 0

        loss = self.crf_layer(probablities, labels_masked, reduction='token_mean', mask = None)

        emissions_ = self.crf_layer.decode( probablities , mask = None)
        emissions = [item for sublist in emissions_ for item in sublist] # flatten the nest list of emissions

        target_emissions = torch.zeros(probablities.shape[0], probablities.shape[1])
        target_emissions = target_emissions.cuda()
        for eachIndex in range( target_emissions.shape[0] ):
            target_emissions[ eachIndex, :torch.tensor( emissions_[eachIndex] ).shape[0] ] = torch.tensor( emissions_[eachIndex] )
        
        return loss, target_emissions, probablities_mask_expanded, labels, mask, mask