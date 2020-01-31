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
stop_words.extend(['like', 'from', 'subject', 're', 'edu', 'use', 'i', 'me', 'my', 
                   'myself', 'we', 'or', 'ors', 'orselves', 'yo', 'yor', 'yors', 
                   'yorself', 'yorselves', 'he', 'him', 'his', 'himself', 'she', 
                   'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
                   'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                   'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                   'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                   'did', 'doing', 'a', 'an', 'the', 'and', 'bt', 'if', 'or', 'becase',
                   'as', 'ntil', 'while', 'of', 'at', 'by', 'for', 'with', 'abot',
                   'against', 'between', 'into', 'throgh', 'dring', 'before', 'after',
                   'above', 'below', 'to', 'from', 'p', 'down', 'in', 'ot', 'on', 
                   'off', 'over', 'nder', 'again', 'frther', 'then', 'once', 'here',
                   'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                   'each', 'few', 'more', 'most', 'other', 'some', 'sch', 'no', 'nor',
                   'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                   't', 'can', 'will', 'jst', 'don', 'shold', 'now', 'know', 'go', 'say',
                   'people', 'want', 'think', 'be','as', 'able', 'about', 'above', 'according',
                   'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 
                   'against', 'aint', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 
                   'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an',
                   'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 
                   'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 
                   'are', 'arent', 'around', 'as', 'aside', 'ask', 'asking', 'associated',
                   'at', 'available', 'away', 'awfully', 'be', 'became', 'because', 'become'
                   , 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being',
                   'believe', 'below', 'beside', 'besides', 'best', 'better', 'between',
                   'beyond', 'both', 'brief', 'but', 'by', 'cmon', 'cs', 'came', 'can',
                   'cant', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 
                   'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 
                   'consequently', 'consider', 'considering', 'contain', 'containing', 
                   'contains', 'corresponding', 'could', 'could', 'course',
                   'currently', 'definitely', 'described', 'despite', 'did', 
                   'didnt', 'different', 'do', 'does', 'does',
                   'doing', 'dont', 'done', 'down', 'downwards', 'during', 'each', 'edu', 
                   'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely',
                   'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody',
                   'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 
                   'except', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following',
                   'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 
                   'further', 'furthermore', 'get', 'gets', 'getting', 'given', 'gives',
                   'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'had', 
                 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter',
                 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him',
                 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however',
                 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 
                 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 
                 'into', 'inward', 'is', 'isnt', 'it', 'its', 'itself', 'just', 'keep', 
                 'keeps', 'kept', 'know', 'knows', 'known', 'last', 'lately', 'later', 
                 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 
                 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'mainly', 'many',
                 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 
                 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'name', 
                 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither',
                 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non',
                 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 
                 'nowhere', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay',
                 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 
                 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 
                 'outside', 'over', 'overall', 'own', 'particular', 'particularly', 
                 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 
                 'probably', 'provides', 'que', 'quite', 'qv', 'rather', 'rd', 're',
                 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively',
                 'respectively', 'right', 'said', 'same', 'saw', 'say', 'saying', 'says', 
                 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 
                 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously',
                'seven', 'several', 'shall', 'she', 'should', 'since', 'six', 'so', 'some',
                'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 
                'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying',
                'still', 'sub', 'such', 'sup', 'sure', 'take', 'taken', 'tell', 'tends',
                'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 
                'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 
                'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to',
                'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'value', 'various', 'very', 'via', 'viz', 'vs', 
                'want', 'wants', 'was', 'we','welcome', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 
                'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', 
                'wonder', 'would', 'would', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'zero'])
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")

#debate_corpus taken from "debate_corpus_script1.py" only using 2015-2016 data
all_text=(debate_corpus)[['text']]
all_text.values.tolist()
data2 = all_text.values.tolist()


all_text_list=
all_text=', '.join(all_text.text)


    df=debate_corpus['text']
    # Convert to list
    data = df.values.tolist()
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in df]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub('\n', '', sent) for sent in data]
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
    dictionary = corpora.Dictionary(data_lemmatized)
        
        # Create Corpus
    texts = data_lemmatized
        
        # Term Document Frequency
    corpus = [id2word.doc2bow(text2) for text2 in texts]
    print(corpus[:2])
 
        # View
    id2word[0]
    # Human readable format of corpus (term-frequency)
    [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:2]]



        # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
           id2word=dictionary,
           num_topics=50, 
           random_state=100,
           update_every=1,
           chunksize=100,
           passes=10,
           alpha='auto',
           per_word_topics=True)
    
    
from gensim import corpora, models
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
pprint(lda_model_tfidf.print_topics())




doc=data[0:1]


for index, score in sorted(lda_model[corpus, key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

# Print the Keyword in the 10 topics
        doc_lda = lda_model_tfidf[corpus[:1]]
        pprint(doc_lda.print_topics())

        top10_tup=lda_model_tfidf.print_topics()[5]
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
       
        #start finalization by entering processed directory for later outputs
    os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\processed\topics")

   
       
        top5=quoted.findall(top10)[0:5]
         
        #start organizing to be put into an output file with date and candidate
        date=df_scraped['date'][0]
        candidate=df_scraped['candidate'][0]
        
        topics_df=pd.DataFrame((date, candidate, top5))
        
        #output to the topics folder
        topics_df.to_csv("top5topics_" + str(candidate) +"_"+ str(date) + ".csv")
        vis_2.to_csv("vis_" + str(candidate) +"_"+ str(date) + ".csv")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#GuidedLDA
import scipy.sparse as ss
from corextopic import corextopic as ct
df=debate_corpus['text']
  # Convert to list
data = df.values.tolist()
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in df]
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub('\n', '', sent) for sent in data]
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
        
# Remove Stop Word
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
print(corpus[:1])




# Define a matrix where rows are samples (docs) and columns are features (words)
X = np.array([[0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)
# Sparse matrices are also supported
X = ss.csr_matrix(X)
# Word labels for each column can be provided to the model
words = ['dog', 'cat', 'fish', 'apple', 'orange']
# Document labels for each row can be provided
docs = ['fruit doc', 'animal doc', 'mixed doc']

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=2)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=words, docs=docs)

topics = topic_model.get_topics()
for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n+1)+': '+','.join(words)
    print(topic_str)

        
from corextopic import vis_topic as vt
vt.vis_rep(topic_model, column_label=words, prefix='topic-model-example')

