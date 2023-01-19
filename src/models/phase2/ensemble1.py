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
from utilities.helper_functions import get_packed_padded_output

class ENSEMBLE1(nn.Module):

    def __init__(self, freeze_bert, tokenizer, model, exp_args):
        super(ENSEMBLE1, self).__init__()

        self.tokenizer = tokenizer

        #Instantiating BERT model object 
        self.transformer_layer = model
        transformer_dim = 768
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.transformer_layer.parameters():
                p.requires_grad = False


        # log reg predictor
        self.hidden2tag = nn.Linear(transformer_dim, exp_args.num_labels)

        if exp_args.predictor == 'crf':
            # crf layer
            self.crf_layer = CRF(exp_args.num_labels, batch_first=True)

        # loss calculation
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None, input_offs=None, mode = None, args = None):

        # Transformer
        outputs = self.transformer_layer(
            input_ids,
            attention_mask = attention_mask
        )
        sequence_output = outputs[0]

        # mask the unimportant tokens before log_reg
        mask = (
            (input_ids != self.tokenizer.pad_token_id)
            & (input_ids != self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token))
            & (input_ids != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )

        mask_expanded = mask.unsqueeze(-1).expand(sequence_output.size())
        sequence_output *= mask_expanded.float()

        labels_masked = labels * mask.long()
        offsets_masked = input_offs * mask.long()

        # linear layer (log reg) to emit class probablities
        probablities = F.relu ( self.hidden2tag( sequence_output ) )
        probablities_mask_expanded = mask.unsqueeze(-1).expand(probablities.size())
        probablities_masked = probablities * probablities_mask_expanded.float()

        if args.predictor == 'linear':

            cumulative_loss = torch.cuda.FloatTensor([0.0]) 
            for i in range(0, probablities.shape[0]):

                loss = self.loss_fct( probablities_masked[i] , labels_masked[i]  )
                cumulative_loss += loss
            
            average_loss = cumulative_loss /  probablities.shape[0]

            return average_loss, probablities, probablities_mask_expanded, labels, mask, mask

        else:

            # on the first time steps XXX CLS token is active at position 0
            for eachIndex in range( mask.shape[0] ):
                mask[eachIndex, 0] = True

            for eachIndex in range( labels_masked.shape[0] ):
                labels_masked[eachIndex, 0] = 0

            # CRF emissions
            loss = self.crf_layer(probablities_masked, labels_masked, reduction='token_mean', mask = None)

            emissions_ = self.crf_layer.decode( probablities_masked , mask = None)
            emissions = [item for sublist in emissions_ for item in sublist] # flatten the nest list of emissions

            target_emissions = torch.zeros(probablities_masked.shape[0], probablities_masked.shape[1])
            target_emissions = target_emissions.cuda()
            for eachIndex in range( target_emissions.shape[0] ):
                target_emissions[ eachIndex, :torch.tensor( emissions_[eachIndex] ).shape[0] ] = torch.tensor( emissions_[eachIndex] )
            
            return loss, target_emissions, probablities_mask_expanded, labels, mask, mask