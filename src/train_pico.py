##################################################################################
# Imports
##################################################################################
# staple imports
from ast import arg
import warnings

from load_pico import fetch_val

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import datetime
import datetime as dt
# Memory leak
import gc
import glob
import json
import logging
import os
import pdb
import random
import shutil
# statistics
import statistics
import sys
import time
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

# sklearn
import sklearn

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, plot_confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# Torch modules
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# Visualization
from tqdm import tqdm

# Transformers 
from transformers import (AdamW, AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer, RobertaConfig,
                          RobertaModel, get_linear_schedule_with_warmup)

warnings.filterwarnings('ignore')

# mlflow 
from utilities.mlflow_logging import *

def printMetrics(cr, args):

    if args.num_labels == 5:   
        return cr['macro avg']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score'], cr['4']['f1-score']
    elif args.num_labels == 4:   
        return cr['macro avg']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score']
    elif args.num_labels == 2:
        return cr['macro avg']['f1-score'], cr['1']['f1-score']


def flattenIt(x):

    return np.asarray(x.cpu(), dtype=np.float32).flatten()

def constrained_beam_search(x, cbs_constraints):

    # if 'cuda' in str(cbs_constraints.device):
    if isinstance(cbs_constraints, np.ndarray) == False:

        temp_list = cbs_constraints.tolist()
        if len(set(temp_list)) > 2:
     
            start_index = cbs_constraints.tolist().index(1)
            end_index = cbs_constraints.tolist().index(2)

            # set all the predictions before start index to 0
            x[0:start_index] = torch.Tensor([0]) * len( x[0:start_index] )

            # Set all the predictions after end index to 0
            x[end_index:] = torch.Tensor([0]) * len( x[end_index:] )

    else:
    
        # Do a constrained beam search only when both start and end indices are present
        if len(set(cbs_constraints)) > 2:
            start_index = list(cbs_constraints).index(1)
            end_index = list(cbs_constraints).index(2)

            # set all the predictions before start index to 0
            x[0:start_index] = [0] * len( x[0:start_index] )

            # Set all the predictions after end index to 0
            x[end_index:] = [0] * len( x[end_index:] )

    return x


def evaluate(defModel, optimizer, scheduler, development_dataloader, exp_args, epoch_number = None, mode=None):
    
    print('The mode is: ', mode)

    mean_acc = 0
    mean_loss = 0
    count = 0
    total_val_loss_coarse = 0

    with torch.no_grad() :
        # collect all the evaluation predictions and ground truth here
        all_GT = []
        all_masks = []
        all_predictions = []
        all_tokens = []

        eval_epochs_logits_coarse_i = np.empty(1, dtype=np.int64)
        eval_epochs_labels_coarse_i = np.empty(1, dtype=np.int64)
        eval_epochs_cbs_coarse_i = np.empty(1, dtype=np.int64)

        class_rep_temp = []

        for e_input_ids_, e_labels, e_input_mask, e_input_pos, e_input_offsets in development_dataloader:

            e_input_ids_ = e_input_ids_.to(f'cuda:{defModel.device_ids[0]}')

            with torch.cuda.device_of(e_input_ids_.data): # why am I cloning this variable?
                e_input_ids = e_input_ids_.clone()

            # load the variables on the device
            e_input_mask = e_input_mask.to(f'cuda:{defModel.device_ids[0]}')
            e_labels = e_labels.to(f'cuda:{defModel.device_ids[0]}')
            e_input_pos = e_input_pos.to(f'cuda:{defModel.device_ids[0]}')
            e_input_offsets = e_input_offsets.to(f'cuda:{defModel.device_ids[0]}')

            # print( e_input_ids.shape )
            # print( e_input_mask.shape )
            # print( e_labels.shape )
            # print( e_input_pos.shape )
            # print( e_input_offsets.shape )

            e_loss, e_output, e_output_masks, e_labels, e_labels_mask, e_mask = defModel(e_input_ids, attention_mask=e_input_mask, labels=e_labels, input_pos=e_input_pos, input_offs=e_input_offsets, mode = mode, args = exp_args) 

            # # shorten the input_ids to match the e_output shape (This is to retrieve the natural langauge words from the input IDs)
            # e_input_ids = e_input_ids[:, :e_output.shape[1]]   

            # if len(list( e_output.shape )) > 1:
            #     if 'crf' not in exp_args.model:
            #         max_probs = torch.max(e_output, dim=2) # get the highest of two probablities
            #         e_logits = max_probs.indices
            #     else:
            #         e_logits = e_output 
            # else: 
            #     e_logits = e_output

            mean_loss += abs( torch.mean(e_loss) ) 

            for i in range(0, e_labels.shape[0]):

                # convert continuous probas to classes for b_output and b_labels
                if 'crf' not in exp_args.model:
                    e_output_class = torch.argmax(e_output[i, ], dim=1)
                else:
                    e_output_class = e_output[i, ]

                if mode == 'test' or exp_args.supervision == 'fs':
                    e_label_class = e_labels[i, ]
                    e_input_offsets_class = e_input_offsets[i, ]
                elif exp_args.supervision == 'ws':
                    e_label_class = torch.argmax(e_labels[i, ], dim=1)

                selected_preds_coarse = torch.masked_select( e_output_class, e_mask[i, ])
                selected_labs_coarse = torch.masked_select( e_label_class, e_mask[i, ])
                selected_offs_coarse = torch.masked_select( e_input_offsets_class, e_mask[i, ])

                if selected_preds_coarse.shape != selected_labs_coarse.shape:
                    print( selected_preds_coarse.shape, ' : ', selected_labs_coarse.shape  )

                selected_preds_coarse = selected_preds_coarse.detach().to("cpu").numpy()
                selected_labs_coarse = selected_labs_coarse.detach().to("cpu").numpy()
                selected_offs_coarse = selected_offs_coarse.detach().to("cpu").numpy()

                # TODO: Run the Constrained Beam Search here. Add an if statement for CBS or not...
                if exp_args.cbs == True:
                    selected_preds_coarse = constrained_beam_search( selected_preds_coarse, selected_offs_coarse )

                eval_epochs_logits_coarse_i = np.append(eval_epochs_logits_coarse_i, selected_preds_coarse, 0)
                eval_epochs_labels_coarse_i = np.append(eval_epochs_labels_coarse_i, selected_labs_coarse, 0)
                eval_epochs_cbs_coarse_i = np.append( eval_epochs_cbs_coarse_i, selected_offs_coarse, 0 )

        #         temp_cr = classification_report(y_pred= masked_preds.cpu(), y_true=masked_labs.cpu(), labels=list(range(exp_args.num_labels)), output_dict=True) 
        #         class_rep_temp.append(temp_cr['macro avg']['f1-score'])

        #         all_masks.extend( e_mask[i, ] )
        #         all_GT.extend( e_labels[i, ] )
        #         all_predictions.extend( e_logits[i, ] )
        #         all_tokens.extend( e_input_ids[i, ] )          

        # avg_val_loss = mean_loss / len(development_dataloader)

        # # stack the list of tensors into a tensor
        # all_masks_tens = torch.stack(( all_masks ))
        # all_GT_tens =  torch.stack(( all_GT ))
        # all_preds_tens = torch.stack(( all_predictions ))
        # all_token_tens = torch.stack(( all_tokens ))

        # # mask the prediction tensor
        # selected_preds_coarse = torch.masked_select( all_preds_tens.to(f'cuda:0'), all_masks_tens )
        # # mask the label tensor
        # selected_labs_coarse = torch.masked_select( all_GT_tens.to(f'cuda:0'), all_masks_tens )
        # # mask the natural language token tensor but with attention mask carefully
        # selected_tokens_coarse = torch.masked_select( all_token_tens.to(f'cuda:0'), all_masks_tens )   

        # # flatten the masked tensors
        # all_pred_flat = flattenIt( selected_preds_coarse )
        # # all_pred_flat = np.asarray(selected_preds_coarse.cpu(), dtype=np.float32).flatten()
        # all_GT_flat = flattenIt( selected_labs_coarse )
        # # all_GT_flat = np.asarray(selected_labs_coarse.cpu(), dtype=np.float32).flatten()
        # all_tokens_flat = flattenIt( selected_tokens_coarse )
        # # all_tokens_flat = np.asarray(selected_tokens_coarse.cpu(), dtype=np.int64).flatten()

        # Final classification report and confusion matrix for each epoch
        # print( len( eval_epochs_logits_coarse_i ) )
        # print( len( eval_epochs_cbs_coarse_i ) )

        val_cr = classification_report(y_pred= eval_epochs_logits_coarse_i, y_true=eval_epochs_labels_coarse_i, labels=list(range(exp_args.num_labels)), output_dict=True, digits=4)

        # confusion_matrix and plot
        labels = [0, 1]
        cm = confusion_matrix(eval_epochs_logits_coarse_i, eval_epochs_labels_coarse_i, labels, normalize=None)

    return val_cr, eval_epochs_logits_coarse_i, eval_epochs_labels_coarse_i, cm        

                                
# Train
def train(defModel, optimizer, scheduler, train_dataloader, development_dataloader, exp_args):

    torch.autograd.set_detect_anomaly(True)

    saved_models = []

    with torch.enable_grad():
        best_f1 = 0.0

        for epoch_i in range(0, exp_args.max_eps):
            # Accumulate loss over an epoch
            total_train_loss = 0

            # accumulate predictions and labels over the epoch
            train_epoch_logits_coarse_i = np.empty(1, dtype=np.int64)
            train_epochs_labels_coarse_i = np.empty(1, dtype=np.int64)

            # Training for all the batches in this epoch
            for step, batch in enumerate(train_dataloader):

                # Clear the gradients
                optimizer.zero_grad()

                b_input_ids = batch[0].to(f'cuda:{defModel.device_ids[0]}')
                b_labels = batch[1].to(f'cuda:{defModel.device_ids[0]}')
                b_masks = batch[2].to(f'cuda:{defModel.device_ids[0]}')
                b_pos = batch[3].to(f'cuda:{defModel.device_ids[0]}')
                b_input_offs = batch[4].to(f'cuda:{defModel.device_ids[0]}')

                b_loss, b_output, b_output_masks, b_labels, b_labels_mask, b_mask = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels, input_pos=b_pos, input_offs = b_input_offs, args = exp_args)                

                total_train_loss += abs( torch.mean(b_loss) ) 

                abs( torch.mean(b_loss) ).backward()

                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

                #Optimization step
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                for i in range(0, b_labels.shape[0]): # masked select excluding the post padding 

                    # convert continuous probas to classes for b_output and b_labels
                    if 'crf' not in exp_args.model:
                        b_output_class = torch.argmax(b_output[i, ], dim=1)
                    else:
                        b_output_class = b_output[i, ]
                    if exp_args.supervision == 'ws':
                        b_label_class = torch.argmax(b_labels[i, ], dim=1)
                    elif exp_args.supervision == 'fs':
                        b_label_class = b_labels[i, ]
                        b_input_offsets_class = b_input_offs[i, ]

                    selected_preds_coarse = torch.masked_select( b_output_class, b_mask[i, ])
                    selected_labs_coarse = torch.masked_select( b_label_class, b_mask[i, ])
                    selected_offs_coarse = torch.masked_select( b_input_offsets_class, b_mask[i, ])

                    selected_preds_coarse = selected_preds_coarse.detach().to("cpu").numpy()
                    selected_labs_coarse = selected_labs_coarse.detach().to("cpu").numpy()
                    selected_offs_coarse = selected_offs_coarse.detach().to("cpu").numpy()

                    # TODO: Run the Constrained Beam Search here. Add an if statement for CBS or not...
                    if exp_args.cbs == True:
                        selected_preds_coarse = constrained_beam_search( selected_preds_coarse, selected_offs_coarse )

                    train_epoch_logits_coarse_i = np.append(train_epoch_logits_coarse_i, selected_preds_coarse, 0)
                    train_epochs_labels_coarse_i = np.append(train_epochs_labels_coarse_i, selected_labs_coarse, 0)

                if step % exp_args.print_every == 0:
                    cr = sklearn.metrics.classification_report(y_pred= train_epoch_logits_coarse_i, y_true= train_epochs_labels_coarse_i, labels= list(range(exp_args.num_labels)), output_dict=True)
                    f1, f1_1 = printMetrics(cr, exp_args)
                    if exp_args.log == True:
                        logMetrics("train macro f1", f1, epoch_i)
                        logMetrics(f"train f1 {exp_args.entity}", f1_1, epoch_i)
                    print('Training: Epoch {} with macro average F1: {}, F1 {}: {}'.format(epoch_i, f1, exp_args.entity, f1_1))


            ## Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            train_cr = classification_report(y_pred= train_epoch_logits_coarse_i, y_true=train_epochs_labels_coarse_i, labels= list(range(exp_args.num_labels)), output_dict=True, digits=4)             

            val_cr, eval_epochs_logits_coarse_i, eval_epochs_labels_coarse_i, cm  = evaluate(defModel, optimizer, scheduler, development_dataloader, exp_args, epoch_i)
           
            val_f1, val_f1_1 = printMetrics(val_cr, exp_args)
            if exp_args.log == True:
                logMetrics("val macro f1", val_f1, epoch_i)
                logMetrics(f"val f1 {exp_args.entity}", val_f1_1, epoch_i)

            string2print = "val f1" + str(exp_args.entity)
            print('Validation: Epoch {} with macro average F1: {}, F1 {}: {}'.format(epoch_i, val_f1, exp_args.entity, val_f1_1))

             # If this is the last epoch then print the classification metrics
            if epoch_i == (exp_args.max_eps - 1):
                print( val_cr['macro avg']['precision'], ',', val_cr['macro avg']['recall'], ',', val_cr['macro avg']['f1-score']
                , ',', val_cr['1']['precision'], ',', val_cr['1']['recall'], ',', val_cr['1']['f1-score']
                , ',', val_cr['0']['precision'], ',', val_cr['0']['recall'], ',', val_cr['0']['f1-score'] )

            # # Process of saving the model
            if val_f1 > best_f1:

            #     if exp_args.supervision == 'ws':
            #         base_path = "/mnt/nas2/results/Results/systematicReview/distant_pico/models/Transformers/" + str(exp_args.supervision) + '/' + str(exp_args.ent) + "/" + str(exp_args.version) + "/" + str(exp_args.exp_level) + "/" + str(exp_args.seed)
            #         if not os.path.exists( base_path ):
            #             oldmask = os.umask(000)
            #             os.makedirs(base_path)
            #             os.umask(oldmask)
            #     elif exp_args.supervision == 'fs':
            #         base_path = "/mnt/nas2/results/Results/systematicReview/distant_pico/models/Transformers/" + str(exp_args.supervision) + '/' + str(exp_args.ent) + "/" + str(exp_args.seed)
            #         if not os.path.exists( base_path ):
            #             oldmask = os.umask(000)
            #             os.makedirs(base_path)
            #             os.umask(oldmask)


                print("Best validation F1 improved from {} to {} ...".format( best_f1, val_f1 ))
                # model_name_here = base_path + '/' + str(exp_args.embed) + '_epoch_' + str(epoch_i) + '.pth'
                # print('Saving the best model for epoch {} with mean F1 score of {} '.format(epoch_i, val_f1 )) 
                # torch.save(defModel.state_dict(), model_name_here/)
                # best_f1 = val_f1
                # saved_models.append(model_name_here)

    return saved_models