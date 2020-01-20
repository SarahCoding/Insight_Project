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
print(tweets.columns.values)
tweets_clean=tweets[['text', 'created_at', 'geo', 'lang', 'place', 'coordinates',
                     'user.followers_count', 'user.location', 'user.geo_enabled',
                     'id', 'favorite_count', 'retweet_count']]

tweets_clean.head
