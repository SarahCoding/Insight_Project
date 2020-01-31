# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:19:07 2020

@author: sarah
"""


#in commandline: twitterscraper
import twitterscraper
import pandas as pd
import os,json
import glob

#get in the directory of the jsons first
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\Debates\Tweets_Debate_20191220")

#scrape twitter
#def scrapetwitter(topics, candidate, date):
 #twitterscraper("bernie sanders near:Boston" -bd 2016-02-01 -ed 2016-03-01 -o boston_bernie_20160201_0301.json -l 10000)


#this finds our jsons
path_to_json=(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\Debates\Tweets_Debate_20191220")
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)

filelist = os.listdir(path_to_json) 
list = []
frame = pd.DataFrame()

for file in json_files:
    tweet_df = pd.read_json(file)
    state = file[:3]
    tweet_df['State'] = state
    #timestamp=df2['timestamp'][0]
    date1=file[:12]
    date=date1[4:]
    tweet_df['Date'] = date
    candidate=file[13:-12]
    tweet_df['Candidate']=candidate
    list.append(tweet_df)
concat_jsons = pd.concat(list)

concat_jsons.columns
#I define the pandas dataframe with the columns of interest extracted from an example json
Tweets_Debate_20191220 = concat_jsons[['State','Date', 'Candidate', 'tweet_id',
                                 'text', 'hashtags', 'user_id', 'is_replied', 'likes', 
                                 'replies', 'retweets']]

    os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
    Tweets_Debate_20191220.to_csv("Tweets_Debate_20191220" + ".csv")  



    
