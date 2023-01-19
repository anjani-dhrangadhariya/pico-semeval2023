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
                             confusion_matrix, f1_score, 
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

def re_stitch_tokens(tokens, labels, subtoken_dummy = 100):

    re_stitched = []
    labels = labels.tolist()

    for i, (t,l) in enumerate(zip(tokens, labels)):
        if i != len(tokens):
            if l != subtoken_dummy:
                re_stitched.append( t )
            elif l == subtoken_dummy:
                stitched_token = re_stitched[i-1] + str(t)
                re_stitched.append( stitched_token )

                for x in range(1, 20):
                    if labels[i-x] == subtoken_dummy:
                        re_stitched[i-x] = stitched_token
                    else:
                        re_stitched[i-x] = stitched_token
                        break
        else:
            re_stitched.append( t )

    assert len(tokens) == len(labels) == len(re_stitched)

    return np.array(re_stitched)


def write_preds(input, preds, labs):

    write_dir = '/mnt/nas2/results/Results/systematicReview/SemEval2023/predictions'
    # replace_func = np.vectorize(lambda x: x.replace('','-'))
    # input = replace_func(input)

    write_np = np.column_stack([input, labs, preds])
    np.savetxt(f'{write_dir}/inspect_best.tsv', write_np, delimiter=';', fmt='%s')

    return None

def printMetrics(cr, labels):

    if labels == 5:   
        return tuple( [ cr['macro avg']['f1-score'], cr['0']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score'], cr['4']['f1-score'] ] )
    elif labels == 4:   
        return tuple( [ cr['macro avg']['f1-score'], cr['0']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score'] ] )
    elif labels == 2:
        return tuple( [ cr['macro avg']['f1-score'], cr['0']['f1-score'], cr['1']['f1-score'] ] )

def print_last_epoch(cr, num_labels):

    # print the metrics of the last epoch
    if num_labels == 2:
        print( round(cr['macro avg']['precision'], 4), ',', round(cr['macro avg']['recall'], 4), ',', round(cr['macro avg']['f1-score'], 4)
        , ',', round(cr['0']['precision'], 4), ',', round(cr['0']['recall'], 4), ',', round(cr['0']['f1-score'], 4)
        , ',', round(cr['1']['precision'], 4), ',', round(cr['1']['recall'], 4), ',', round(cr['1']['f1-score'], 4)
        )

    elif num_labels == 4:
        print( round(cr['macro avg']['precision'], 4), ',', round(cr['macro avg']['recall'], 4), ',', round(cr['macro avg']['f1-score'], 4)
        , ',', round(cr['0']['precision'], 4), ',', round(cr['0']['recall'], 4), ',', round(cr['0']['f1-score'], 4)
        , ',', round(cr['1']['precision'], 4), ',', round(cr['1']['recall'], 4), ',', round(cr['1']['f1-score'], 4)
        , ',', round(cr['2']['precision'], 4), ',', round(cr['2']['recall'], 4), ',', round(cr['2']['f1-score'], 4)
        , ',', round(cr['3']['precision'], 4), ',', round(cr['3']['recall'], 4), ',', round(cr['3']['f1-score'], 4)
        )


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


def evaluate(defModel, defTokenizer, optimizer, scheduler, development_dataloader, exp_args, epoch_number = None, mode=None):
    
    if mode == 'test':
        print('The mode is: ', mode)

    mean_acc = 0
    mean_loss = 0
    count = 0
    total_val_loss_coarse = 0
    total_val_loss_fine = 0

    with torch.no_grad() :
        # collect all the evaluation predictions and ground truth here
        all_GT = []
        all_masks = []
        all_predictions = []
        all_tokens = []

        all_predictions_fine = []
        all_GT_fine = []

        eval_epochs_logits_coarse_i = np.empty(1, dtype=np.int64)
        eval_epochs_labels_coarse_i = np.empty(1, dtype=np.int64)
        eval_epochs_cbs_coarse_i = np.empty(1, dtype=np.int64)
        eval_epochs_inputs_coarse_i = np.empty(1, dtype=np.int64)

        class_rep_temp = []

        # for e_input_ids_, e_labels, e_input_mask, e_input_pos, e_input_offsets in development_dataloader:
        for e_batch in development_dataloader:

            e_input_ids_ = e_batch[0].to(f'cuda:{defModel.device_ids[0]}')

            with torch.cuda.device_of(e_input_ids_.data): # why am I cloning this variable?
                e_input_ids = e_input_ids_.clone()

            # load the variables on the device
            e_labels = e_batch[1].to(f'cuda:{defModel.device_ids[0]}')
            e_labels_fine = e_batch[2].to(f'cuda:{defModel.device_ids[0]}')
            e_input_mask = e_batch[3].to(f'cuda:{defModel.device_ids[0]}')
            e_input_pos = e_batch[4].to(f'cuda:{defModel.device_ids[0]}')
            e_input_offsets = e_batch[5].to(f'cuda:{defModel.device_ids[0]}')

            e_input_char = e_batch[6].to(f'cuda:{defModel.device_ids[0]}')
            e_input_ortho = e_batch[7].to(f'cuda:{defModel.device_ids[0]}')

            e_loss_coarse, e_output, e_labels, e_loss_fine, e_output_fine, e_labels_fine, e_mask, e_cumulative_loss = defModel(e_input_ids, attention_mask=e_input_mask, labels=e_labels, labels_fine=e_labels_fine, input_pos=e_input_pos, input_offs=e_input_offsets, input_char_encode=e_input_char, input_char_ortho=e_input_ortho, mode = mode, args = exp_args) 

            mean_loss += e_cumulative_loss
            total_val_loss_coarse += e_loss_coarse
            total_val_loss_fine += e_loss_fine


            for i in range(0, e_labels.shape[0]):

                selected_preds = torch.masked_select(e_output[i, ].cuda(), e_mask[i, ])
                selected_labs = torch.masked_select(e_labels[i, ].cuda(), e_mask[i, ])

                e_cr = classification_report(y_pred=selected_preds.to("cpu").numpy(), y_true=selected_labs.to("cpu").numpy(), labels=list(range(exp_args.num_labels)) , output_dict=True)

                all_predictions.extend(selected_preds.to("cpu").numpy())
                all_GT.extend(selected_labs.to("cpu").numpy())

                ###############################################################################
                selected_f_preds = torch.masked_select(e_output_fine[i, ].cuda(), e_mask[i, ])
                selected_f_labs = torch.masked_select(e_labels_fine[i, ].cuda(), e_mask[i, ]) 

                e_cr_fine = classification_report(y_pred=selected_f_preds.to("cpu").numpy(), y_true=selected_f_labs.to("cpu").numpy(), labels=list(range(exp_args.num_labels*2)) , output_dict=True)
                
                all_predictions_fine.extend(selected_f_preds.to("cpu").numpy())
                all_GT_fine.extend(selected_f_labs.to("cpu").numpy())             

            count += 1

        avg_val_loss = mean_loss / len(development_dataloader)        
        avg_val_loss_coarse = total_val_loss_coarse / len(development_dataloader)     
        avg_val_loss_fine = total_val_loss_fine / len(development_dataloader)     

        # Final classification report and confusion matrix for each epoch
        all_pred_flat_ = np.asarray(all_predictions).flatten()
        all_GT_flat_ = np.asarray(all_GT).flatten()
        val_cr = classification_report(y_pred=all_pred_flat_, y_true=all_GT_flat_, labels=list(range(exp_args.num_labels)), output_dict=True)
        # print(val_cr)

        all_pred_flat_fine = np.asarray(all_predictions_fine).flatten()
        all_GT_flat_fine = np.asarray(all_GT_fine).flatten()
        val_cr_fine = classification_report(y_pred=all_pred_flat_fine, y_true=all_GT_flat_fine, labels=list(range(exp_args.num_labels*2)), output_dict=True)
        # print(val_cr_fine)

    return mean_loss / count, val_cr, val_cr_fine, all_pred_flat_, all_GT_flat_, all_pred_flat_fine, all_GT_flat_fine       

                                
# Train
def train_mtl(defModel, defTokenizer, optimizer, scheduler, train_dataloader, development_dataloader, exp_args):

    torch.autograd.set_detect_anomaly(True)

    saved_models = []

    with torch.enable_grad():
        best_f1 = 0.0

        for epoch_i in range(0, exp_args.max_eps):
            # Accumulate loss over an epoch
            total_train_loss = 0
            total_train_loss_coarse = 0
            total_train_loss_fine = 0

            # (coarse-grained) accumulate predictions and labels over the epoch
            train_epoch_logits = []
            train_epochs_labels = []
            # (fine-grained) accumulate predictions and labels over the epoch
            train_epoch_logits_fine = []
            train_epochs_labels_fine = []

            # Training for all the batches in this epoch
            for step, batch in enumerate(train_dataloader):

                # Clear the gradients
                optimizer.zero_grad()

                b_input_ids = batch[0].to(f'cuda:{defModel.device_ids[0]}')
                b_labels = batch[1].to(f'cuda:{defModel.device_ids[0]}')
                b_labels_fine = batch[2].to(f'cuda:{defModel.device_ids[0]}')
                b_masks = batch[3].to(f'cuda:{defModel.device_ids[0]}')
                b_pos = batch[4].to(f'cuda:{defModel.device_ids[0]}')
                b_input_offs = batch[5].to(f'cuda:{defModel.device_ids[0]}')

                b_input_char = batch[6].to(f'cuda:{defModel.device_ids[0]}')
                b_input_ortho = batch[7].to(f'cuda:{defModel.device_ids[0]}')

                b_loss_coarse, b_output, b_labels, b_loss_fine, b_output_fine, b_f_labels, b_mask, cumulative_loss = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels, labels_fine=b_labels_fine, input_pos=b_pos, input_offs = b_input_offs, input_char_encode=b_input_char, input_char_ortho=b_input_ortho, args = exp_args)                

                total_train_loss += cumulative_loss
                total_train_loss_coarse += b_loss_coarse
                total_train_loss_fine += b_loss_fine

                cumulative_loss.backward()

                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

                #Optimization step
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                for i in range(0, b_labels.shape[0]): 

                    selected_preds = torch.masked_select(b_output[i, ].cuda(), b_mask[i, ])
                    selected_labs = torch.masked_select(b_labels[i, ].cuda(), b_mask[i, ])

                    train_epoch_logits.extend(selected_preds.to("cpu").numpy())
                    train_epochs_labels.extend(selected_labs.to("cpu").numpy())

                    selected_preds_fine = torch.masked_select(b_output_fine[i, ].cuda(), b_mask[i, ])
                    selected_labs_fine = torch.masked_select(b_f_labels[i, ].cuda(), b_mask[i, ])

                    train_epoch_logits_fine.extend( selected_preds_fine.to('cpu').numpy() )
                    train_epochs_labels_fine.extend( selected_labs_fine.to("cpu").numpy() )

                # if step % exp_args.print_every == 0:
                #     cr = classification_report(y_pred= train_epoch_logits, y_true=train_epochs_labels, labels= list(range(exp_args.num_labels)), output_dict=True)
                #     f1 = printMetrics(cr, 2)
                #     print( 'Training: Epoch {} with macro average F1: {}, F1 (0): {}, F1 (1): {}'.format( epoch_i, f1[0], f1[1], f1[2] ) )
                    
                #     cr_fine = classification_report(y_pred= train_epoch_logits_fine, y_true=train_epochs_labels_fine, labels= list(range(exp_args.num_labels*2)), output_dict=True)
                #     f1_fine = printMetrics(cr_fine, 4)
                #     print( 'Training: Epoch {} with macro average F1 (fine): {}, F1 (0): {}, F1 (1): {}, F1 (2): {}, F1 (3): {}'.format( epoch_i, f1_fine[0], f1_fine[1], f1_fine[2], f1_fine[3], f1_fine[4] ) )

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            print("  Average training loss: {0:.6f}".format(avg_train_loss))

            avg_train_loss_coarse = total_train_loss_coarse / len(train_dataloader)

            avg_train_loss_fine = total_train_loss_fine / len(train_dataloader)

            train_cr = classification_report(y_pred= train_epoch_logits, y_true=train_epochs_labels, labels= list(range(exp_args.num_labels)), output_dict=True) 
            train_f1 = printMetrics(train_cr, 2)
            print( 'Training: Epoch {} with macro average F1: {}, F1 (0): {}, F1 (1): {}'.format( epoch_i, train_f1[0], train_f1[1], train_f1[2] ) )

            train_cr_fine = classification_report(y_pred= train_epoch_logits_fine, y_true=train_epochs_labels_fine, labels= list(range(exp_args.num_labels*2)), output_dict=True) 
            train_f1_fine = printMetrics(train_cr_fine, 4)
            print( 'Training: Epoch {} with macro average F1 (fine): {}, F1 (0): {}, F1 (1): {}, F1 (2): {}, F1 (3): {}'.format( epoch_i, train_f1_fine[0], train_f1_fine[1], train_f1_fine[2], train_f1_fine[3], train_f1_fine[4] ) )

            val_loss, val_cr, val_cr_fine, all_pred_flat_, all_GT_flat_, all_pred_flat_fine, all_GT_flat_fine = evaluate(defModel, defTokenizer, optimizer, scheduler, development_dataloader, exp_args, epoch_i)
           
            val_f1 = printMetrics(val_cr, 2)
            print( 'Validation: Epoch {} with macro average F1: {}, F1 (0): {}, F1 (1): {}'.format( epoch_i, val_f1[0], val_f1[1], val_f1[2] ) )
            val_f1_fine = printMetrics(val_cr_fine, 4)
            print( 'Validation: Epoch {} with macro average F1 (fine): {}, F1 (0): {}, F1 (1): {}, F1 (2): {}, F1 (3): {}'.format( epoch_i, val_f1_fine[0], val_f1_fine[1], val_f1_fine[2], val_f1_fine[3], val_f1_fine[4] ) )

            #  If this is the last epoch then print the classification metrics
            if epoch_i == (exp_args.max_eps - 1):
                print_last_epoch(val_cr, 2)
                print_last_epoch(val_cr_fine, 4)

            # # Process of saving the model
            if val_f1[0] > best_f1:

                if exp_args.supervision == 'fs':
                    base_path = "/mnt/nas2/results/Results/systematicReview/SemEval2023/models/"
                    if not os.path.exists( base_path ):
                        oldmask = os.umask(000)
                        os.makedirs(base_path)
                        os.umask(oldmask)


                print("Best validation F1 improved from {} to {} ...".format( best_f1, val_f1[0] ))
                model_name_here = base_path + '/' + str(exp_args.entity) + str(exp_args.embed) + '_epoch_' + str(epoch_i) + '.pth'
                print('Saving the best model for epoch {} with mean F1 score of {} '.format(epoch_i, val_f1[0] )) 
                torch.save(defModel.state_dict(), model_name_here)
                best_f1 = val_f1[0]
                saved_models.append(model_name_here)

    return saved_models