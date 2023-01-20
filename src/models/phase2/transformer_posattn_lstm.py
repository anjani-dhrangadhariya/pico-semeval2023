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
import warnings
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

class TRANSFORMERPOSAttenLin(nn.Module):

    def __init__(self, freeze_bert, tokenizer, model, exp_args):
        super(TRANSFORMERPOSAttenLin, self).__init__()
        #Instantiating BERT model object 
        self.transformer_layer = model
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.transformer_layer.parameters():
                p.requires_grad = False

        self.hidden_dim = 512
        self.hidden_pos = 36
        self.tokenizer = tokenizer

        self.self_attention = nn.MultiheadAttention(768, 1, bias=True) # attention mechanism from PyTorch

        # lstm layer for POS tags and the embeddings and attention mechanism
        if exp_args.bidrec == True:  
            self.lstmpos_layer = nn.LSTM(input_size=18, hidden_size = self.hidden_pos, num_layers = 1, bidirectional=exp_args.bidrec, batch_first=True)
            self.self_attention_pos = nn.MultiheadAttention(self.hidden_pos * 2, 1, bias=True) # attention mechanism from PyTorch

            self.lstm_layer = nn.LSTM(input_size=768, hidden_size = self.hidden_dim, num_layers = 1, bidirectional=exp_args.bidrec, batch_first=True)
            self.hidden2tag = nn.Linear(1024, 2)

        elif exp_args.bidrec == False:            
            self.lstmpos_layer = nn.LSTM(input_size=18, hidden_size = self.hidden_pos, num_layers = 1, bidirectional=exp_args.bidrec, batch_first=True)
            self.self_attention_pos = nn.MultiheadAttention(self.hidden_pos, 1, bias=True)             

            self.lstm_layer = nn.LSTM(input_size=768, hidden_size = self.hidden_dim, num_layers = 1, bidirectional=exp_args.bidrec, batch_first=True)
            self.hidden2tag = nn.Linear(512, 2)

        # loss calculation
        self.loss_fct = nn.CrossEntropyLoss()

    
    def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None, input_offs=None, mode = None, args = None):

        # Transformer
        outputs = self.transformer_layer(
            input_ids,
            attention_mask = attention_mask
        )

        # output 0 = batch size 6, tokens MAX_LEN, each token dimension 768 [CLS] token
        sequence_output = outputs[0]
        print( 'Output of transformers: ', sequence_output.shape )

        # Attention weighted transformer output output
        attention_applied, attention_weights = self.self_attention( sequence_output, sequence_output, sequence_output, need_weights=True, attn_mask = None )
        print( 'Output of transformers after attention applied: ', attention_applied.shape )

        # ------------------------------ POS input preprocessing ------------------------------------
        # one hot encode POS tags
        input_pos = input_pos
        print( 'Shape of input POS tags: ', input_pos.shape )
        packed_pos_input, pos_perm_idx, pos_seq_lengths, total_length_pos = get_packed_padded_output_dataparallel(input_pos.float(), attention_mask)
        print( 'Shape of input packed POS tags: ', packed_pos_input[0].shape )
        self.lstmpos_layer.flatten_parameters()
        packed_pos_output, (ht_pos, ct_pos) = self.lstmpos_layer(packed_pos_input)
        print( 'LSTM output of POS tags: ', packed_pos_output[0].shape)

        # Unpack and reorder the output
        # pos_output, pos_input_sizes = pad_packed_sequence(packed_pos_output, batch_first=True, total_length=total_length_pos)
        # _, unperm_idx = pos_perm_idx.sort(0)
        # lstm_pos_output = pos_output[unperm_idx]
        # seq_lengths_ordered = pos_seq_lengths[unperm_idx]
        # target = torch.zeros(lstm_pos_output.shape[0], input_pos.shape[1], lstm_pos_output.shape[2])
        # target = target.cuda()

        # target[:, :lstm_pos_output.shape[1], :] = lstm_pos_output # Expand dimensions of the LSTM transformed pos embeddings

        # Attention over the POS embeddings to up-weigh important features
        # attention_applied_pos, attention_weights_pos = self.self_attention_pos( target, target, target, key_padding_mask=None, need_weights=True, attn_mask = None )

        # concatenate attention weighted Transformer embeddings and attention weighted POS tags
        concatenatedVectors = torch.cat( (attention_applied, input_pos), 2) # concatenate at dim 2 for embeddings and tags
        concatenatedVectors = concatenatedVectors.cuda()

        # lstm with masks (same as attention masks) applied on concatenated embeddings + pos features
        lstm_packed_input, perm_idx, seq_lengths = get_packed_padded_output(attention_applied, attention_mask, self.tokenizer)
        print( len( lstm_packed_input ) )
        print( lstm_packed_input[0].shape )
        lstm_packed_output, (ht, ct) = self.lstm_layer(lstm_packed_input)

        # Unpack and reorder the output
        lstm_output, input_sizes = pad_packed_sequence(lstm_packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output[unperm_idx] # lstm_output.shape = shorter than the padded torch.Size([6, 388, 512])
        seq_lengths_ordered = seq_lengths[unperm_idx]
        
        # shorten the labels as per the batchsize
        labels = labels[:, :lstm_output.shape[1]]

        # mask the unimportant tokens before log_reg (NOTE: CLS token (position 0) is not masked!!!)
        mask = (
            (input_ids[:, :lstm_output.shape[1]] != self.tokenizer.pad_token_id)
            & (input_ids[:, :lstm_output.shape[1]] != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )

        mask_expanded = mask.unsqueeze(-1).expand(lstm_output.size())
        lstm_output *= mask_expanded.float()
        labels *= mask.long()

        # log reg
        probablities = F.relu ( self.hidden2tag( lstm_output ) )
        cumulative_loss = torch.cuda.FloatTensor([0.0]) 

        for i in range(0, probablities.shape[0]):
            loss = self.loss_fct( probablities[i] , labels[i]  )
            cumulative_loss += loss
        
        average_loss = cumulative_loss /  probablities.shape[0]

        return average_loss, probablities, mask_expanded, labels, mask, mask