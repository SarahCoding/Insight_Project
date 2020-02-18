# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 06:59:54 2020 @author: sarah"""


from lxml import html
import lxml
import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import os

import re
import numpy as np
import pandas as pd
from pprint import pprint

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




###TOPIC MODELLING 
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
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

#get debate corpus
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
debate_corpus=pd.read_csv("debate_corpus_40.csv")
print(debate_corpus.participants.unique())
debate_corpus.head()
debate_corpus.loc[32]

# Convert to list
data = debate_corpus.text.values.tolist()

# Remove at signs and emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub('\n+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

print(data[:1])

#Tokenize words and clean up the text
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])

# Build the bigram and trigram models
#bigrams are two words that occur frequently together and trigrams are 3 words

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
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[39:40])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
id2word[0]
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

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

for index, score in sorted(lda_model[corpus[32]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

def format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus, texts=data_text):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus, texts=documents['text'])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(32)
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\models\topic_output")
df_dominant_topic_notext = df_dominant_topic[df_dominant_topic.columns[0:3]]
df_dominant_topic_info=debate_corpus[debate_corpus.columns[1:3]]
df_dominant_topic_notext_full = pd.concat([df_dominant_topic_notext, df_dominant_topic_info], axis=1, sort=False)

                                 
                                   
df_dominant_topic_notext_full.to_csv("Dominant_topic_in_doc_bow_lda_notext.csv")


# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics
df_dominant_topics.to_csv("Dominant_topics_bow_lda.csv")



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
        model=lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized[32:33], start=2, limit=40, step=6)



# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()



















