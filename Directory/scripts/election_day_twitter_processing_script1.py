# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
cwd = os.getcwd()
cwd
import pandas as pd

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#Study data
tweets=pd.read_csv('election_day_tweets.csv')
tweets.head
tweets.columns
tweets_clean=tweets[['text', 'created_at', 'geo', 'lang', 'place', 
                     'coordinates', 'user.followers_count', 
                     'user.location', 'user.geo_enabled', 
                     'id', 'favorite_count', 'retweet_count']]

tweets_clean.columns
tweets_clean.head

#Using only geo enabled tweets
geo_true=tweets['user.geo_enabled']==True
tweets_geo=tweets_clean[geo_true]
isalabama=tweets_geo['user.location']=='Alabama'
alabama_tweets=tweets_geo[alabama_tweets]
