# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 06:59:54 2020 @author: sarah"""


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
from joblib import dump
from gensim.models import CoherenceModel

from gensim import corpora, models
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

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
                'Julian', 'Castro', 'Tim', 'Biden', 'Paul', 'Congressman', 'David',
                'buttigieg', 'pete', 'klobuchar', 'warren','yang','booker', 'elizabeth', 'lester',
                'holt', 'jose', 'anderson', 'harris', 'kamala', 'andrew', 'chuck', 'michael',
                'lindsey', 'david', 'cori', 'vice', 'pence', 'harri', 'harris','ronald','reagan',
                'tulsi', 'gabbard', 'julian', 'tapper', 'rachel', 'maddow', 'marco',
                'graham', 'nancy', 'nanci', 'carson', 'anderson', 'cooper', 'congresswoman',
                'congressman'])


os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")

#get debate corpus
debate_corpus=pd.read_csv("debate_corpus_40.csv")
print(debate_corpus.participants.unique())
debate_corpus.head()
debate_corpus.loc[32]

# Convert to list
data = debate_corpus.text.values.tolist()

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

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


doc_sample = documents[33:].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


processed_docs = documents['text'].map(preprocess)
print(processed_docs)
processed_docs_nostopwords = remove_stopwords(processed_docs)
print(processed_docs_nostopwords)


dictionary = gensim.corpora.Dictionary(processed_docs_nostopwords)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs_nostopwords]
bow_corpus[32]

bow_doc_32 = bow_corpus[32]
for i in range(len(bow_doc_32)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_32[i][0], 
                                               dictionary[bow_doc_32[i][0]], 
bow_doc_32[i][1]))


#back to the model
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
    pprint(doc)
    break



#BOW MODEL
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=25, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


for index, score in sorted(lda_model[bow_corpus[33]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))



#TFIDIF MODEL
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=dictionary, passes=2, workers=4)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


for index, score in sorted(lda_model_tfidf[bow_corpus[32]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))




#Save model   
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\models")
dump(lda_model, "LDA_MODEL_40DebateCorpus.joblib")    
dump(lda_model_tfidf, "TFIDF-LDA_MODEL_40DebateCorpus.joblib")    

os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\models\topic_output")
f = open("bow_lda_topics.txt", "a")#saving to an output file
#Print topics for bow lda for each debate
for debate_number in range(56):
    print("This is debate number:", debate_number, file=f)
    for index, score in sorted(lda_model[bow_corpus[debate_number]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 20)), file=f)
f.close()

f2 = open("tfidf_lda_topics.txt", "a")
#Print topics for tfidf lda for each debate
for debate_number in range(56):
    print("This is debate number:", debate_number, file=f2)
    for index, score in sorted(lda_model[bow_corpus[debate_number]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 20)), file=f2)
f2.close()



#unseen_document = 'How a Pentagon deal became an identity crisis for Google'
#bow_vector = dictionary.doc2bow(preprocess(unseen_document))
#for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
 #   print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))




#Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


#tfidf
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model=lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=dictionary, passes=2, workers=4)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_tfidf, texts=processed_docs, start=2, limit=40, step=6)



# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()




#bow lda
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model=lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dictionary, passes=2, workers=2)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=processed_docs, start=2, limit=40, step=6)



# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()










