# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:19:07 2020

@author: sarah
"""


#in commandline: twitterscraper
import pandas as pd
import os,json
import glob
#this finds our jsons
path_to_json=(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\States_tweets_2016_02_01-06_08_2k")
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)

filelist = os.listdir(path_to_json) 
list = []
frame = pd.DataFrame()
for file in filelist:
    df2 = pd.read_json(file)
    state = file[0:-32]
    df2['State'] = state
    list.append(df2)
concat_jsons = pd.concat(list)

#I define the pandas dataframe with the columns of interest extracted from an example json
df_clean = concat_jsons[['State','timestamp', 'tweet_id',
             'user_id', 'is_replied', 'likes', 
             'replies', 'retweets', 'text', 'hashtags']]

df_clean.to_csv(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned\States_tweets_2016_02_01-06_08_2k.csv")



    
