# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:11:26 2020

@author: sarah
"""


from lxml import html
import lxml
import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import os
import csv



import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
from gensim import corpora, models
import joblib
from joblib import dump
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Visualize the topics
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\models")
lda_model=joblib.load("LDA_MODEL_40DebateCorpus.joblib")    

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
