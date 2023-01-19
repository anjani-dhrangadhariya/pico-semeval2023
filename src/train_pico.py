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

import numpy as np
import pandas as pd

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
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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


def write_preds(input, preds, labs, exp_args):

    base_path = '/mnt/nas2/results/Results/systematicReview/SemEval2023/predictions'
    file_name_here = base_path + '/' + str(exp_args.entity) + '/' + str(exp_args.seed) + '/' + str(exp_args.embed) + '/' + str(exp_args.model) + '_' + str(exp_args.predictor) + '_ep_' + str(exp_args.max_eps - 1) + '.tsv'

    write_np = np.column_stack([input, labs, preds])
    np.savetxt(file_name_here, write_np, delimiter=';', fmt='%s')

    return None

def plot_cm(cm, exp_args):

    base_path = '/home/anjani/pico-semeval2023/src/visualization/phase2/cm'
    file_name_here = base_path + '/' + str(exp_args.entity) + '/' + str(exp_args.seed) + '/' + str(exp_args.embed) + '/' + str(exp_args.model) + '_' + str(exp_args.predictor) + '_ep_' + str(exp_args.max_eps - 1) + '.png'

    # Plot confusion matrix
    # plt.figure(figsize = (10,7))
    # sns.heatmap(cm, annot=True, fmt="d")
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
    fig.savefig(file_name_here)

    return None

def printMetrics(cr, args):

    if args.num_labels == 5:   
        return tuple( [ cr['macro avg']['f1-score'], cr['0']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score'], cr['4']['f1-score'] ] )
    elif args.num_labels == 4:   
        return tuple( [ cr['macro avg']['f1-score'], cr['0']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score'] ] )
    elif args.num_labels == 2:
        return tuple( [ cr['macro avg']['f1-score'], cr['0']['f1-score'], cr['1']['f1-score'] ] )

def print_last_epoch(cr, args):

    # print the metrics of the last epoch
    if args.num_labels == 2:
        print( round(cr['macro avg']['precision'], 4)*100 , ',', round(cr['macro avg']['recall'], 4)*100, ',', round(cr['macro avg']['f1-score'], 4)*100
        , ',', round(cr['0']['precision'], 4)*100, ',', round(cr['0']['recall'], 4)*100, ',', round(cr['0']['f1-score'], 4)*100
        , ',', round(cr['1']['precision'], 4)*100, ',', round(cr['1']['recall'], 4)*100, ',', round(cr['1']['f1-score'], 4)*100
        )

    elif args.num_labels == 4:
        print( round(cr['macro avg']['precision'], 4)*100, ',', round(cr['macro avg']['recall'], 4), ',', round(cr['macro avg']['f1-score'], 4)
        , ',', round(cr['0']['precision'], 4)*100, ',', round(cr['0']['recall'], 4)*100, ',', round(cr['0']['f1-score'], 4)*100
        , ',', round(cr['1']['precision'], 4)*100, ',', round(cr['1']['recall'], 4)*100, ',', round(cr['1']['f1-score'], 4)*100
        , ',', round(cr['2']['precision'], 4)*100, ',', round(cr['2']['recall'], 4)*100, ',', round(cr['2']['f1-score'], 4)*100
        , ',', round(cr['3']['precision'], 4)*100, ',', round(cr['3']['recall'], 4)*100, ',', round(cr['3']['f1-score'], 4)*100
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

    with torch.no_grad() :
        # collect all the evaluation predictions and ground truth here
        all_GT = []
        all_masks = []
        all_predictions = []
        all_tokens = []

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
            e_input_mask = e_batch[2].to(f'cuda:{defModel.device_ids[0]}')
            e_input_pos = e_batch[3].to(f'cuda:{defModel.device_ids[0]}')
            e_input_offsets = e_batch[4].to(f'cuda:{defModel.device_ids[0]}')
            e_input_char = e_batch[5].to(f'cuda:{defModel.device_ids[0]}')
            e_input_ortho = e_batch[6].to(f'cuda:{defModel.device_ids[0]}')

            # print( e_input_ids.shape )
            # print( e_input_mask.shape )
            # print( e_labels.shape )
            # print( e_input_pos.shape )
            # print( e_input_offsets.shape )

            e_loss, e_output, e_output_masks, e_labels, e_labels_mask, e_mask = defModel(e_input_ids, attention_mask=e_input_mask, labels=e_labels, input_pos=e_input_pos, input_offs=e_input_offsets, mode = mode, args = exp_args) 

            mean_loss += abs( torch.mean(e_loss) ) 

            for i in range(0, e_labels.shape[0]):

                # convert continuous probas to classes for b_output and b_labels
                if 'crf' not in exp_args.predictor:
                    e_output_class = torch.argmax(e_output[i, ], dim=1)
                else:
                    e_output_class = e_output[i, ]

                e_label_class = e_labels[i, ]
                e_input_offsets_class = e_input_offsets[i, ]
                e_input_ids_class = e_input_ids_[i, ]
                    

                selected_preds_coarse = torch.masked_select( e_output_class, e_mask[i, ])
                selected_preds_coarse = selected_preds_coarse.detach().to("cpu").numpy()

                selected_offs_coarse = torch.masked_select( e_input_offsets_class, e_mask[i, ])
                selected_offs_coarse = selected_offs_coarse.detach().to("cpu").numpy()                

                # TODO: before masking, convert the ids to tokens
                original_tokens = defTokenizer.convert_ids_to_tokens(e_input_ids_class)
                # use e_label_class to re-stitch the tokens
                e_input_ids_class = re_stitch_tokens(original_tokens, e_label_class)
                selected_inputs_coarse = e_input_ids_class[e_mask[i, ].detach().to("cpu").numpy()   ]

                selected_labs_coarse = torch.masked_select( e_label_class, e_mask[i, ])
                selected_labs_coarse = selected_labs_coarse.detach().to("cpu").numpy()

                # TODO: Run the Constrained Beam Search here. Add an if statement for CBS or not...
                if exp_args.cbs == True:
                    selected_preds_coarse = constrained_beam_search( selected_preds_coarse, selected_offs_coarse )

                eval_epochs_logits_coarse_i = np.append(eval_epochs_logits_coarse_i, selected_preds_coarse, 0)
                eval_epochs_labels_coarse_i = np.append(eval_epochs_labels_coarse_i, selected_labs_coarse, 0)
                eval_epochs_cbs_coarse_i = np.append( eval_epochs_cbs_coarse_i, selected_offs_coarse, 0 )
                eval_epochs_inputs_coarse_i = np.append( eval_epochs_inputs_coarse_i, selected_inputs_coarse, 0 )


        val_cr = classification_report(y_pred= eval_epochs_logits_coarse_i, y_true=eval_epochs_labels_coarse_i, labels=list(range(exp_args.num_labels)), output_dict=True, digits=4)

        # confusion_matrix and plot
        labels = list( range( exp_args.num_labels ) )
        print( labels )
        cm = sklearn.metrics.confusion_matrix(eval_epochs_logits_coarse_i, eval_epochs_labels_coarse_i, labels=labels, normalize=None)

        # Write input IDs and labels down to a file for inspection
        if epoch_number == (exp_args.max_eps - 1):
            write_preds(eval_epochs_inputs_coarse_i, eval_epochs_logits_coarse_i, eval_epochs_labels_coarse_i, exp_args)
            plot_cm(cm, exp_args)


    return val_cr, eval_epochs_logits_coarse_i, eval_epochs_labels_coarse_i, cm        

                                
# Train
def train(defModel, defTokenizer, optimizer, scheduler, train_dataloader, development_dataloader, exp_args):

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
                # print( torch.unique(b_labels) )
                b_masks = batch[2].to(f'cuda:{defModel.device_ids[0]}')
                b_pos = batch[3].to(f'cuda:{defModel.device_ids[0]}')
                b_input_offs = batch[4].to(f'cuda:{defModel.device_ids[0]}')

                b_input_char = batch[5].to(f'cuda:{defModel.device_ids[0]}')
                b_input_ortho = batch[6].to(f'cuda:{defModel.device_ids[0]}')

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
                    if 'crf' not in exp_args.predictor:
                        b_output_class = torch.argmax(b_output[i, ], dim=1)
                    else:
                        b_output_class = b_output[i, ]

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
                    f1 = printMetrics(cr, exp_args)
                    # if exp_args.log == True:
                        # logMetrics("train macro f1", f1, epoch_i)
                        # logMetrics(f"train f1 {exp_args.entity}", f1_1, epoch_i)
                    if exp_args.num_labels == 2:
                        print( 'Training: Epoch {} with macro average F1: {}, F1 (0): {}, F1 (1): {}'.format( epoch_i, f1[0], f1[1], f1[2] ) )
                    if exp_args.num_labels == 4:
                        print( 'Training: Epoch {} with macro average F1: {}, F1 (0): {}, F1 (1): {}, F1 (2): {}, F1 (3): {}'.format( epoch_i, f1[0], f1[1], f1[2], f1[3], f1[4] ) )


            ## Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            train_cr = classification_report(y_pred= train_epoch_logits_coarse_i, y_true=train_epochs_labels_coarse_i, labels= list(range(exp_args.num_labels)), output_dict=True, digits=4)             

            val_cr, eval_epochs_logits_coarse_i, eval_epochs_labels_coarse_i, cm  = evaluate(defModel, defTokenizer, optimizer, scheduler, development_dataloader, exp_args, epoch_i)
           
            val_f1 = printMetrics(val_cr, exp_args)
            # if exp_args.log == True:
            #     logMetrics("val macro f1", val_f1, epoch_i)
            #     logMetrics(f"val f1 {exp_args.entity}", val_f1_1, epoch_i)
            if exp_args.num_labels == 2:
                print( 'Validation: Epoch {} with macro average F1: {}, F1 (0): {}, F1 (1): {}'.format( epoch_i, val_f1[0], val_f1[1], val_f1[2] ) )
            if exp_args.num_labels == 4:
                print( 'Validation: Epoch {} with macro average F1: {}, F1 (0): {}, F1 (1): {}, F1 (2): {}, F1 (3): {}'.format( epoch_i, val_f1[0], val_f1[1], val_f1[2], val_f1[3], val_f1[4] ) )

             # If this is the last epoch then print the classification metrics
            if epoch_i == (exp_args.max_eps - 1):
                print_last_epoch(val_cr, exp_args)

            # # Process of saving the model
            if val_f1[0] > best_f1:

                if exp_args.supervision == 'fs':
                    base_path = "/mnt/nas2/results/Results/systematicReview/SemEval2023/models/"
                    if not os.path.exists( base_path ):
                        oldmask = os.umask(000)
                        os.makedirs(base_path)
                        os.umask(oldmask)


                print("Best validation F1 improved from {} to {} ...".format( best_f1, val_f1[0] ))
                model_name_here = base_path + '/' + str(exp_args.entity) + '/' + str(exp_args.seed) + '/' + str(exp_args.embed) + '/' + str(exp_args.model) + '_' + str(exp_args.predictor) + '_ep_' + str(epoch_i) + '.pth'
                print('Saving the best model for epoch {} with mean F1 score of {} '.format(epoch_i, val_f1[0] )) 
                torch.save(defModel.state_dict(), model_name_here)
                best_f1 = val_f1[0]
                saved_models.append(model_name_here)

    return saved_models