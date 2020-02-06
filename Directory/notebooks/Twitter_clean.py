# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:40:43 2020

@author: sarah
"""


#in commandline: twitterscraper
import twitterscraper
import pandas as pd
import os,json
import glob

os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped")

#scrape twitter
#def scrapetwitter(topics, candidate, date):
#os.system #goes into terminal to scrape tweets
 #   twitterscraper("bernie sanders near:Boston" -bd 2016-02-01 -ed 2016-03-01 -o boston_bernie_20160201_0301.json -l 10000)


#this finds our jsons
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\topic_tweets")
path_to_json=(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\topic_tweets")
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)

filelist = os.listdir(path_to_json) 
list = []
frame = pd.DataFrame()
for file in filelist:
    df2 = pd.read_json(file)
    state = file[8:-35]
    df2['State'] = state
    timestamp=df2['timestamp'][0]
    list.append(df2)
concat_jsons = pd.concat(list)

#I define the pandas dataframe with the columns of interest extracted from an example json
tweet_data_clean = concat_jsons[['State','timestamp', 'tweet_id',
             'user_id', 'is_replied', 'likes', 
             'replies', 'retweets', 'text', 'hashtags']]

    os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
    tweet_data_clean.to_csv("tweet_data_clean_" + str(state)+ "_"+str(candidate_name) +"_"+ ".csv")  


