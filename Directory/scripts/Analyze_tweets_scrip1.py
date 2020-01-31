# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:46:05 2020

@author: sarah
"""


#in commandline: twitterscraper
import twitterscraper
import pandas as pd
import os,json
import glob
import re
#get in the directory of the jsons first
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
Tweets_Debate_20191220 = pd.read_csv("Tweets_Debate_20191220.csv")
# race, color, guns 

#topics obtained from LDA model of the specific debate
#Topic: 0.119*"biden" + 0.083*"harri" + 0.066*"vice" + 0.021*"speaker" + 0.013*"okay" 
#+ 0.009*"option" + 0.009*"michigan" + 0.008*"farmer" + 0.008*"fossil" + 0.008*"racial" 
#+ 0.007*"attorney" + 0.007*"bold" + 0.007*"measur" + 0.006*"babi" + 0.006*"color" + 
#0.006*"carbon" + 0.006*"teacher" + 0.006*"congressman" + 0.006*"divers" + 0.006*"assault"

topicsraw='0.119*"biden" + 0.083*"harri" + 0.066*"vice" + 0.021*"speaker" + 0.013*"okay" + 0.009*"option" + 0.009*"michigan" + 0.008*"farmer" + 0.008*"fossil" + 0.008*"racial" + 0.007*"attorney" + 0.007*"bold" + 0.007*"measur" + 0.006*"babi" + 0.006*"color" + 0.006*"carbon" + 0.006*"teacher" + 0.006*"congressman" + 0.006*"divers" + 0.006*"assault"'
topicslist=re.findall('"([^"]*)"', topicsraw)
topics=','.join(topicslist)

#NEED TO FIND A WAY TO ACCOUNT FOR 0 MATCHING TWEETS AS NO OUTPUT FILE IS CREATED

tweets=Tweets_Debate_20191220["text"]
#tokenize tweets
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

tw_tok=[]
for tweet in range(len(tweets)):
    temptweet=tknzr.tokenize(tweets[tweet])
    temptweet1=','.join(temptweet)
    tw_tok.append(temptweet1)

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)

get_cosine_sim(topics, tw_tok[2])[1][0]


tw_cos=[]
for tweet in range(len(tw_tok)):
    cos=get_cosine_sim(topics, tw_tok[tweet])[1][0]
    tw_cos.append(cos)

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

