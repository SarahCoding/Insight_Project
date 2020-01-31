# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 06:59:54 2020 @author: sarah"""


from lxml import html
import lxml
import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import os


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)


os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw")
latest_links=pd.read_csv("links_to_latest_debates.csv")[0:].values.tolist()
latest_links=pd.read_csv("links_to_latest_debates.csv").values.tolist()
# Function to convert   
# Python program to convert a list 
# to string using list comprehension   
links2=[]
for link_number in range(8):
    links2.append(latest_links[link_number][0])
    

#debate_website_test3="https://www.rev.com/blog/transcripts/democratic-debate-transcript-houston-september-12-2019"
def debate_scraper_latest(debate_website):
    #test=(debate_website)
    #return test  
    page2=urllib.request.urlopen(debate_website) #will used specified url
    soup_debate2=BeautifulSoup(page2,'html.parser')
    # Take out the <div> of name and get its value
    date=soup_debate2.find('div', attrs=('class', 'fl-rich-text')).text
    debate_text=soup_debate2.find('div', attrs=('class', 'fl-callout-text')).text
    participants=soup_debate2.find('div', attrs=('class', 'fl-module fl-module-rich-text fl-node-5e186cbbd29ea')).text
    #Create a data frame of the soup outputs for future nlp
    debate_data=pd.DataFrame(([[date, participants, debate_text]]))
    debate_data.columns=['date', 'participants', 'text']
    return debate_data

debate_corpus_latest=pd.DataFrame()
for debate_website in links2:
    debate_data2=debate_scraper_latest(debate_website)
    debate_corpus_latest=pd.DataFrame(debate_corpus_latest.append(debate_data2))




#data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = debate_corpus_latest[['text']]
#data_text['index'] = data_text.index
documents = data_text

import nltk

from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
#print(WordNetLemmatizer().lemmatize('went', pos='v'))
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[1:].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['text'].map(preprocess)
processed_docs[:10]


dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[30]

bow_doc_30 = bow_corpus[30]

for i in range(len(bow_doc_30)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_30[i][0], 
                                                     dictionary[bow_doc_30[i][0]], 
                                                     bow_doc_30[i][1]))



from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break

#lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
#for idx, topic in lda_model.print_topics(-1):
 #   print('Topic: {} \nWords: {}'.format(idx, topic))

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

                
for index, score in sorted(lda_model[bow_corpus[debate_number]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

topics_final=[]

for debate_number in range(40):
    print("This is debate number:", debate_number)
    for index, score in sorted(lda_model_tfidf[bow_corpus[debate_number]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))






