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
from transformer_pos_crf import TRANSFORMERPOSCRF
from mtl_0 import MTL_0
from mtl_baseline import MTLBASELINE
from mtl_2 import MTL_2
from mtl_3 import MTL_3
from mtl_4 import MTL_4
from mtl_5 import MTL_5
from ensemble1 import ENSEMBLE1
from ensemble2 import ENSEMBLE2
from ensemble3 import ENSEMBLE3
from ensemble5 import ENSEMBLE5
from ensemble7 import ENSEMBLE7
from ensemble9 import ENSEMBLE9
from ensemble11 import ENSEMBLE11
from ensemble13 import ENSEMBLE13


import model_tokenizer

TOKENIZERS = model_tokenizer.HF_TOKENIZERS
MODELS = model_tokenizer.HF_MODELS
MODEL_DICT = { 'ensemble1': ENSEMBLE1, 'ensemble2': ENSEMBLE2, 'ensemble3': ENSEMBLE3, 'ensemble4': ENSEMBLE3, 'ensemble5': ENSEMBLE5, 'ensemble6': ENSEMBLE5, 'ensemble7': ENSEMBLE7, 'ensemble8': ENSEMBLE7, 'ensemble9': ENSEMBLE9, 'ensemble10': ENSEMBLE9, 'ensemble11': ENSEMBLE11, 'ensemble12': ENSEMBLE11, 'ensemble13': ENSEMBLE13, 'ensemble14': ENSEMBLE13, 'transformerlinear': TRANSFORMERLINEAR, 'transformerpos': TRANSFORMERPOS, 'transformerposcrf':TRANSFORMERPOSCRF , 'transformercrf': TRANSFORMERCRF, 'transformerlstmlinear':TRANSFORMERBiLSTMLINEAR, 'transformerlstmattnlin':TRANSFORMERAttenLin, 'transformerposlstmattnlin':TRANSFORMERPOSAttenLin, 'mtl_0':MTL_0, 'mtl_baseline':MTLBASELINE, 'mtl_2':MTL_2, 'mtl_3':MTL_3, 'mtl_4': MTL_4, 'mtl_5': MTL_5 }

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

    if chosen_model in MODEL_DICT:
        load_model = MODEL_DICT[chosen_model]
        model = load_model(args.freeze_bert, tokenizer, modelembed, args)
    else:
        print( 'Provide one of the names from this list...', ', '.join( list(MODEL_DICT.keys() )) )

    return model