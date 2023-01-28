import argparse
from pathlib import Path

# pyTorch essentials
import torch

##################################################################################
# set up the GPU
##################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('Number of GPUs identified: ', n_gpu)
print('You are using ', torch.cuda.get_device_name(0), ' : ', device , ' device')

def getArguments():

    # List of arguments to set up experiment
    parser = argparse.ArgumentParser()

    parser.add_argument('-use_lemma', type = bool, default= False)
    parser.add_argument('-log', type = bool, default= False)

    parser.add_argument('-supervision', type = str, default = 'fs') # label_type = {fs, ws, hs, ...} 
    parser.add_argument('-train_from_scratch', type = str, default=True)

    parser.add_argument('-entity', type = str, default = 'participant') # participant, intervention, outcome, all, all_sep
    parser.add_argument('-num_labels', type = int, default = 2) # 2 for binary (O-span vs. P/I/O) classification, 4 for multiclass (PICO) classification
    parser.add_argument('-data_dir', type = Path, default = '/mnt/nas2/data/systematicReview/semeval2023/data/preprocessed')

    parser.add_argument('-label_type', type = str, default = 'seq_lab') # label_type = {seq_lab, BIO, BIOES, ...} 

    parser.add_argument('-max_len', type = int, default=512)
    parser.add_argument('-seed', type = int, default= 100)
    parser.add_argument('-embed', type = str, default = 'none') # embed = {bioredditbert, biomedroberta, roberta, scibert, bert, biobert, pubmedbert, BioLinkBERT ...} 
    parser.add_argument('-model', type = str, default = 'ensemble7') # model = {transformerpos, transformerposcrf, transformerlinear, transformercrf, transformerlstmlinear, transformerlstmattnlin, transformerposlstmattnlin} 
    parser.add_argument('-predictor', type=str, default = 'none') # { linear, crf }
    parser.add_argument('-pos_encoding', type=str, default='onehot') # onehot, lstm 
    parser.add_argument('-cbs', type=bool, default=False) # Constrained Beam Search 
    parser.add_argument('-bidrec', type = str, default=True)


    parser.add_argument('-print_every', type = int, default= 100)
    parser.add_argument('-mode', type = str, default= "train")

    parser.add_argument('-max_eps', type = int, default= 1)
    parser.add_argument('-batch', type = int, default= 10)
    parser.add_argument('-loss', type = str, default = 'general')
    parser.add_argument('-freeze_bert', action='store_false') # store_false = won't freeze BERT
    parser.add_argument('-lr', type = float, default= 5e-4)
    parser.add_argument('-eps', type = float, default= 1e-8)
    parser.add_argument('-lr_warmup', type = float, default= 0.1) 


    parser.add_argument('-parallel', type = str, default = 'false') # false = won't use data parallel
    parser.add_argument('-gpu', type = int, default = device)

    args = parser.parse_args()


    return args
