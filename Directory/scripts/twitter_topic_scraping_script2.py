# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:19:07 2020

@author: sarah
"""


#in commandline: twitterscraper
import pandas as pd
import os,json
import glob

os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\topic_tweets")


#this finds our jsons
path_to_json=(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\topic_tweets")
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)

filelist = os.listdir(path_to_json) 
list = []
frame = pd.DataFrame()

#tweet_df=pd.read_json("tweets_american_sanders_20160201_20160207.json")

#for more than one file list
for file in filelist:
    tweet_df = pd.read_json(file)
    topic = file[7:-30] #should not be hardcoded
    tweet_df['Topic'] = topic
    list.append(tweet_df)
    
concat_jsons = pd.concat(list)

#I define the pandas dataframe with the columns of interest extracted from an example json
df_clean = concat_jsons[['Topic','timestamp', 'tweet_id',
             'user_id', 'is_replied', 'likes', 
             'replies', 'retweets', 'text', 'hashtags']]

df_clean.to_csv(r"bernie_topics.csv")



    
