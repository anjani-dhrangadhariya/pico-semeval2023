# Transformers 
import sys


from transformers import (AutoConfig, AutoModel,
                          AutoModelForTokenClassification, AutoModelWithLMHead,
                          AutoTokenizer)

path = '/home/anjani/pico-semeval2023/src/models/phase2'
sys.path.append(path)
from models.phase2 import transformer_crf, transformer_linear
from transformer_crf import TRANSFORMERCRF
from transformer_linear import TRANSFORMERLINEAR

import model_tokenizer

TOKENIZERS = model_tokenizer.HF_TOKENIZERS
MODELS = model_tokenizer.HF_MODELS

##################################################################################
# Load the chosen tokenizer
##################################################################################
def choose_tokenizer_type(pretrained_model):

    if pretrained_model in TOKENIZERS and pretrained_model in MODELS:
        tokenizer_ = AutoTokenizer.from_pretrained( TOKENIZERS[pretrained_model] )
        model_ = AutoModel.from_pretrained( MODELS[pretrained_model] )
    else:
        print('Please input a valid model name...')
    

    return tokenizer_ , model_


def choose_model(vector_type, tokenizer, modelembed, chosen_model, args):

    if chosen_model == 'transformerlinear':
        model = TRANSFORMERLINEAR(args.freeze_bert, tokenizer, modelembed, args)
    if chosen_model == 'transformercrf':
        model = TRANSFORMERCRF(args.freeze_bert, tokenizer, modelembed, args)

    return model
