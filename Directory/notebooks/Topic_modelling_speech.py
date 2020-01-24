# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:33:38 2020

@author: sarah
"""

#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
#import proccessing libraries for dataframes
import csv
import pandas as pd
import re
import numpy as np
import pandas as pd
from pprint import pprint
import openpyxl
from openpyxl import load_workbook
import xlrd
import os
import re
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words to remove common words from language
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['like', 'from', 'subject', 're', 'edu', 'use'])

# Import Dataset that will extract topics of interest
# this uses a news website with 20 topics 
# (might be irrellvenat but a good start)

os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped")

#input_df=("speech_scraped_test.csv")
#df_scraped=pd.read_csv(input_df) 


def topic_analysis(filename):  
    df_scraped=pd.read_csv(filename) 
    df_scraped_clean = df_scraped.replace('\n',' ', regex=True)
    print(df_scraped_clean)
    df=df_scraped_clean['text']
    print(df_scraped.head)
    # Convert to list
    data = df.values.tolist()
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in df]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    #data = [re.sub('\n', '', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in df]
    pprint(data[:1])
    
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    
    data_words = list(sent_to_words(data))
    
    print(data_words[:1])
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])
    
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    #nlp = spacy.load('en', disable=['parser', 'ner'])
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    print(data_lemmatized[:1])
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    
    # Create Corpus
    texts = data_lemmatized
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # View
    print(corpus[:1])
    id2word[0]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
       id2word=id2word,
       num_topics=20, 
       random_state=100,
       update_every=1,
       chunksize=100,
       passes=10,
       alpha='auto',
       per_word_topics=True)
    
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    top10_tup=lda_model.print_topics()[1]
    print(top10_tup)
    top10=top10_tup[1]
    quoted = re.compile('"[^"]*"')
    
    top5=quoted.findall(top10)[0:5]
    
    #start organizing to be put into an output file with date and candidate
    date=df_scraped['date'][0]
    candidate=df_scraped['candidate'][0]
    
    topics_df=pd.DataFrame((date, candidate, top5))
    
    #output to the topics folder
    os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\processed\topics")
    topics_df.to_csv("top5topics_" + str(candidate) +"_"+ str(date) + ".csv")
    return topics_df

