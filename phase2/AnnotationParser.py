# imports

import pandas as pd
import ast
import collections
import re
import spacy
import scispacy

# Load tokenizers

#loading the scispacy model
nlp = spacy.load('en_core_sci_sm')

