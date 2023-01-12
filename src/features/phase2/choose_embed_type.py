# Transformers 
import sys
path = '/home/anjani/pico-semeval2023/src/models/phase2'
sys.path.append(path)


from transformers import (AutoConfig, AutoModel,
                          AutoModelForTokenClassification, AutoModelWithLMHead,
                          AutoTokenizer)


from models.phase2 import transformer_crf, transformer_linear
from transformer_crf import TRANSFORMERCRF
from transformer_linear import TRANSFORMERLINEAR
from transformer_lstm_linear import TRANSFORMERBiLSTMLINEAR
from transformer_lstmatten_linear import TRANSFORMERAttenLin
from transformer_posattn_lstm import TRANSFORMERPOSAttenLin
from transformer_pos import TRANSFORMERPOS


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

    if chosen_model == 'transformerpos':
        model = TRANSFORMERPOS(args.freeze_bert, tokenizer, modelembed, args)

    if chosen_model == 'transformercrf':
        model = TRANSFORMERCRF(args.freeze_bert, tokenizer, modelembed, args)

    if chosen_model == 'transformerlstmlinear':
        model = TRANSFORMERBiLSTMLINEAR(args.freeze_bert, tokenizer, modelembed, args)

    if chosen_model == 'transformerlstmattnlin':
        model = TRANSFORMERAttenLin(args.freeze_bert, tokenizer, modelembed, args)
        
    if chosen_model == 'transformerposlstmattnlin':
        model = TRANSFORMERPOSAttenLin(args.freeze_bert, tokenizer, modelembed, args)

    return model
