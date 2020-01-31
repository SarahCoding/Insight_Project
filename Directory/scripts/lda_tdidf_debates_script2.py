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
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
from gensim import corpora, models


#Collecting all the websites needed to build a corpus of debates
website="https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-candidates-debates-1960-2016"
#query the website and return html
page=urllib.request.urlopen(website) #will used specified url
#parse the html using BS
soup=BeautifulSoup(page,'html.parser')
# Creating a list of debate links for future corpus
links=[]
for link in soup.find('tbody').find_all('a'):
    links.append(link.get('href'))
links=links[1:33] #chose debates from present until 2015
#now parsing through the debate links, individually
def debate_scraper(debate_website):
    #test=(debate_website)
    #return test  
    global debate_data
    page=urllib.request.urlopen(debate_website) #will used specified url
    soup_debate=BeautifulSoup(page,'html.parser')
    # Take out the <div> of name and get its value
    debate_main=soup_debate.find('div', attrs=('class', 'wrapper')).find('div', attrs=('class', 'field-docs-content'))
    date=soup_debate.find('div', attrs=('class', "field-docs-start-date-time")).text.split("\n")[1]
    debate_text=debate_main.text
    participants=debate_main.find('p').text.split("\n")[1:]
    
    #Create a data frame of the soup outputs for future nlp
    debate_data=pd.DataFrame(([[date, participants, debate_text]]))
    debate_data.columns=['date', 'participants', 'text']
    return debate_data
debate_corpus=pd.DataFrame()
for debate_website in links:
    debate_data=debate_scraper(debate_website)
    debate_corpus=pd.DataFrame(debate_corpus.append(debate_data))
    debate_corpus['text']=debate_corpus['text'].str.replace(',', '')
    debate_corpus['text']=debate_corpus['text'].str.replace('.', '')
    debate_corpus['date']=debate_corpus['date'].str.replace(',', '')

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

debate_corpus=pd.DataFrame(debate_corpus.append(debate_corpus_latest))

###TOPIC MODELLING 
stop_words = stopwords.words('english')
stop_words.extend(['Have','kasich', 'cruz', 'like', 'from', 'subject', 're', 'edu', 'use', 'I',
                   'me', 'my', 'I\'ve', 'we', 'people', 'I', 'It', 'Those', 'those','we\'re','I\'ve', 'them', 'they', ' I\'m', 'thank', 'you', 'by', 'of',                  'myself', 'we', 'or', 'ors', 'orselves', 'able', 'your', 'yours',
                   'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she','her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
                   'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                   'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                   'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                   'did', 'doing', 'a', 'an', 'the', 'and', 'bt', 'if', 'or', 'becase',
                   'as', 'ntil', 'while', 'of', 'at', 'by', 'for', 'with', 'abot',              'against', 'between', 'into', 'throgh', 'dring', 'before', 'after',
                   'above', 'below', 'to', 'from', 'p', 'down', 'in', 'ot', 'on',                  'off', 'over', 'nder', 'again', 'frther', 'then', 'once', 'here',
                   'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',                   'each', 'few', 'more', 'most', 'other', 'some', 'sch', 'no', 'nor',
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
                'want', 'wants', 'was','We','we','welcome', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 
                'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', 
                'wonder', 'would', 'would\'ve', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'zero',
                'make', 'didn\'t', 'It\'s','It','look','Look', 'This','We\'ll','And','I\'ve', 
                'I\'m', 'Donald', 'Trump', 'Amy', 'Klobuchar:', 'Secondly,', 'I’m',
                'Julian', 'Castro', 'Tim', 'Biden', 'Paul', 'Congressman', 'David'])




#data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = debate_corpus[['text']]
#data_text['index'] = data_text.index
documents = data_text


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

#doc_sample
doc_sample = documents[39:].values[0][0]
words = doc_sample.split()
print(words)
words = [w for w in words if w not in stop_words]
print(words)
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#back to the model
processed_docs = documents['text'].map(preprocess)
print(processed_docs)
processed_docs_nostopwords = [w for w in processed_docs if w not in stop_words]
print(processed_docs_nostopwords)

dictionary = gensim.corpora.Dictionary(processed_docs_nostopwords)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

#sample
bow_doc_30 = bow_corpus[30]
for i in range(len(bow_doc_30)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_30[i][0], 
                                                     dictionary[bow_doc_30[i][0]], 
                                                     bow_doc_30[i][1]))
#back to model
#lda_model
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    
for debate_number in range(40):
    print("This is debate number:", debate_number)
    for index, score in sorted(lda_model[bow_corpus[debate_number]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 20)))
        
with open('out.txt', 'w') as f:
    print >> f, 'Filename:', filename
    print('Filename:', filename, file=f) 


#tfidf
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    print(doc)
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


topics_final=[]

for debate_number in range(40):
    print("This is debate number:", debate_number)
    for index, score in sorted(lda_model_tfidf[bow_corpus[debate_number]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

doc_sample = documents[1:].values[0][0]
bow_vector = dictionary.doc2bow(preprocess(doc_sample))

for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))

# Remove Stop Words
data_words_nostops_sample = remove_stopwords(doc_sample)
bow_vector_2 = dictionary.doc2bow(preprocess(data_words_nostops_sample))


for index, score in sorted(lda_model_tfidf[bow_vector_2], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))


sampledata=data_text.iloc[39,:1]


