# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
#Study data
twitter_data=pd.read_csv(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned\States_tweets_2016_02_01-06_08_2k.csv", dtype={'text': str})
twitter_data.text.apply(str)
twitter_data.hashtags.apply(str)
twitter_data.info()
#twitter_data["text2"]= twitter_data.text.astype(str) 


#tweets=twitter_data['text']

def text_processing(tweets):
    
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweets):
        tweet_blob = TextBlob(tweets)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweets)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweets):
        tweet_list = [ele for ele in tweets.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
        
    return normalization(no_punc_tweet)

twitter_data['tweet_list'] = twitter_data['text'].apply(text_processing)
twitter_data['hashtags_list'] = twitter_data['hashtags'].apply(text_processing)

# number of times each tweet appears
counts = twitter_data.groupby(['text']).size()\
           .reset_index(name='counts')\
           .counts

# define bins for histogram
my_bins = np.arange(0,counts.max()+2, 1)-0.5

# plot histogram of tweet counts
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('copies of each tweet')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()












