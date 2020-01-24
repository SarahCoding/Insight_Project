# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:33:38 2020

@author: sarah
"""

#import libraries for speech scraping
from datetime import date
from lxml import html
import requests
import urllib.request
from bs4 import BeautifulSoup

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
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words to remove common words from language
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['like', 'from', 'subject', 're', 'edu', 'use'])
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#website="https://www.presidency.ucsb.edu/documents/remarks-des-moines-following-the-iowa-caucus-1"

#Scrape website for speech
 
def speech_scraper(website):  
    page=urllib.request.urlopen(website) #will used specified url
    #parse the html using BS
    soup=BeautifulSoup(page,'html.parser')
    soup
    # Take out the <div> of name and get its value
    candidate=soup.find('div', attrs=('class', 'field-docs-person'))
    candidate=candidate.text
    candidate1=candidate.split("\n\n\n \n\n")[1]
    candidate2=candidate.split("\n\n\n\n")[0]
    candidate3=candidate2.split("\n\n\n \n\n")[1]
    candidate_name=candidate3.split("\n\n\n")[0]
    
    date=soup.find('span', attrs=('class', 'date-display-single'))
    date=date.text
    
    text=soup.find('div', attrs=('class', 'field-docs-content'))
    text=text.text
    
    #Create a data frame of the soup outputs for future nlp
    speechdata=pd.DataFrame(([[candidate_name, date, text]]))
    speechdata.columns=['candidate', 'date', 'text']
    
    os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
    speechdata.to_csv("speechdata_" + str(candidate_name) +"_"+ str(date) + ".csv")  
    filename=("speechdata_" + str(candidate_name) +"_"+ str(date) + ".csv")
   
    def topic_analysis(filename):
        df_scraped=pd.read_csv(filename) 
        df_scraped_clean = df_scraped.replace('\n',' ', regex=True)
        df_scraped_clean.to_csv("speechdata_" + str(candidate_name) +"_"+ str(date) + ".csv")  
      
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
        
        
        #start finalization by entering processed directory for later outputs
        os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\processed\topics")

        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        top10_tup=lda_model.print_topics()[5]
        print(top10_tup)
        top10=top10_tup[1]
        quoted = re.compile('"[^"]*"')
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        
        
        #Visualize terms and frequencies
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        vis_all=vis[1]
        vis_top10=vis_all[0:10]
        
        vis_Data = vis_all.groupby("Term")
        vis_Data.describe().head()
        vis_Data.mean().sort_values(by="Freq",ascending=False).head()
        vis_Data.size().sort_values(ascending=False).plot.bar()
        plt.figure(figsize=(15,10))
        vis_Data.size().sort_values(ascending=False).plot.bar()
        plt.xticks(rotation=50)
        #plt.scatter('Term', 'Freq', data=vis_top10)
        plt.xlabel('Term')
        plt.ylabel('Frequency')
        Fig1=plt.gcf()
        plt.show()
        Fig1.savefig("vis_" + str(candidate) +"_"+ str(date) + ".png", bbox_inches='tight')

       
        top5=quoted.findall(top10)[0:5]
         
        #start organizing to be put into an output file with date and candidate
        date=df_scraped['date'][0]
        candidate=df_scraped['candidate'][0]
        
        topics_df=pd.DataFrame((date, candidate, top5))
        
        #output to the topics folder
        topics_df.to_csv("top5topics_" + str(candidate) +"_"+ str(date) + ".csv")
        vis_2.to_csv("vis_" + str(candidate) +"_"+ str(date) + ".csv")
    
    
    


