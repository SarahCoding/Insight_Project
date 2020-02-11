# -*- coding: utf-8 -*-
"""Created on Wed Jan 29 13:46:05 2020 @author: sarah"""

#in commandline: twitterscraper
import twitterscraper
import pandas as pd
import os,json
import glob
import re

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#preprocess topics
#Score: 0.99    
topicsraw1='0.119*"biden" + 0.083*"harri" + 0.066*"vice" + 0.021*"speaker" + 0.013*"okay" + 0.009*"option" + 0.009*"michigan" + 0.008*"farmer" + 0.008*"fossil" + 0.008*"racial" + 0.007*"attorney" + 0.007*"bold" + 0.007*"measur" + 0.006*"babi" + 0.006*"color" + 0.006*"carbon" + 0.006*"teacher" + 0.006*"congressman" + 0.006*"divers" + 0.006*"assault"'
#Score: 0.87
topicsraw2='Topic: 0.105*"biden" + 0.057*"harri" + 0.055*"vice" + 0.017*"speaker" + 0.009*"color" + 0.009*"okay" + 0.009*"attorney" + 0.008*"option" + 0.008*"pharmaceut" + 0.008*"bold" + 0.008*"racial" + 0.008*"nato" + 0.007*"michigan" + 0.007*"farmer" + 0.007*"data" + 0.007*"fossil" + 0.006*"measur" + 0.006*"abort" + 0.006*"carbon" + 0.006*"prescript"'
#Score: 0.13
topicsraw3='Topic: 0.091*"biden" + 0.081*"harri" + 0.053*"vice" + 0.021*"teacher" + 0.019*"congressman" + 0.018*"speaker" + 0.015*"david" + 0.014*"okay" + 0.012*"assault" + 0.009*"michigan" + 0.009*"option" + 0.007*"fossil" + 0.007*"farmer" + 0.007*"racism" + 0.007*"bold" + 0.007*"racial" + 0.006*"divers" + 0.006*"latino" + 0.005*"undocu" + 0.005*"abort"'

topicslist=[topicsraw1, topicsraw2, topicsraw3]
topics_cleaned=[]
for number in range(len(topicslist)):
    topicsraw=re.findall('"([^"]*)"', topicslist[number])
    topics=' '.join(topicsraw)
    topics_cleaned.append(topics)

#ACCOUNT FOR 0 MATCHING TWEETS AS NO OUTPUT FILE IS CREATED

#preprocess tweets
#get in the directory of the jsons first
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
Tweets_Debate_20191220 = pd.read_csv("Tweets_Debate_20191220.csv")
tweets=Tweets_Debate_20191220["text"]
#tokenize tweets
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

tw_tok=[]
for tweet in range(len(tweets)):
    temptweet=tknzr.tokenize(tweets[tweet])
    temptweet1=' '.join(temptweet)
    tw_tok.append(temptweet1)


def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)

get_cosine_sim('blah, blah, blah', 'blah blah blah')

#obtain cos simalirity for each topic key words
tw_cos1=[]
for tweet in range(len(tw_tok)):
    cos=get_cosine_sim(topics_cleaned[0], tw_tok[0])[0][1]
    tw_cos1.append(cos)

tw_cos2=[]
for tweet in range(len(tw_tok)):
    cos2=get_cosine_sim(topics_cleaned[1], tw_tok[tweet])[0][1]
    tw_cos2.append(cos2)

tw_cos3=[]
for tweet in range(len(tw_tok)):
    cos3=get_cosine_sim(topics_cleaned[2], tw_tok[tweet])[0][1]
    tw_cos3.append(cos3)
       
    
    

tw_cos_df=pd.DataFrame(tw_cos)
Tweets_Debate_20191220["cos"]=tw_cos_df

Tweets_Debate_20191220.to_csv("Tweets_Debate_20191220.csv")



#other techniques for similarity

#gun="gun"
#color="color"
#racist="racist"
#Tweets_Debate_20191220["gun"]= Tweets_Debate_20191220["text"].str.find(gun) 

def get_jaccard_sim(str1, str2): 
    a = set(topics.split()) 
    b = set(tweet_example.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
get_jaccard_sim(topics, tw_tok[2])


tw_jac=[]
for tweet in range(len(tw_tok)):
    jacc=get_jaccard_sim(topics, tw_tok[tweet])
    tw_jac.append(jacc)




def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

