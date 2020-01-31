# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:22:50 2020

@author: sarah
"""

os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\topic_tweets")

print("tweet_data_clean_" + str(state)+ "_"+str(candidate_name) +"_"+ ".csv")

twitterscraper "#debates AND race OR guns OR color AND Biden AND Colorado" -bd 2019-12-20 -ed 2019-12-21 -o Col_20191220_Biden_tweets.json -l 10000
