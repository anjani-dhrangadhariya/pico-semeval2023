from transformers import (AutoConfig, AutoModel,
                          AutoModelForTokenClassification, AutoModelWithLMHead,
                          AutoTokenizer)

HF_TOKENIZERS = {
    'bioredditbert': 'cambridgeltl/BioRedditBERT-uncased',
    'bert': 'bert-base-uncased',
    'gpt2': 'gpt2',
    'biobert': 'dmis-lab/biobert-v1.1',
    'scibert': 'allenai/scibert_scivocab_cased',
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'BioLinkBERT':'michiyasunaga/BioLinkBERT-base',
    'biomedroberta': 'allenai/biomed_roberta_base',
    'roberta': 'roberta-base'
    }


HF_MODELS = {
    'bioredditbert': 'cambridgeltl/BioRedditBERT-uncased',
    'bert': 'bert-base-uncased',
    'gpt2': 'gpt2',
    'biobert': 'dmis-lab/biobert-v1.1',
    'scibert': 'allenai/scibert_scivocab_cased',
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'roberta': 'roberta-base', # recognizes at least
    'BioLinkBERT':'michiyasunaga/BioLinkBERT-base', # Kinda good
    'biomedroberta': 'allenai/biomed_roberta_base' # The best one till now
    }
