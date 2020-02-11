# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:40:13 2020

@author: sarah
"""
import twitterscraper
import pandas as pd
import os,json
import glob
from twitterscraper import query_tweets

#os.server("twitterscraper "(#democraticdebate OR #demdebate OR #debates2019 OR debate OR debates OR primary OR primaries OR election OR vote OR #debate OR #debates) AND (#AmyKlobuchar OR Klobuchar OR Amy OR #Klobuchar OR #PeteButtigieg OR #MayorPete OR Pete OR Buttigieg OR #Buttigieg OR #ElizabethWarren OR Elizabeth OR Warren OR #Warren OR #JoeBiden OR Joe OR Biden OR #Biden OR #BernieSanders Bernie OR Sanders OR #Sanders OR #AndrewYang OR Andrew OR Yang OR #Yang OR #TomSteyer OR Tom OR Steyer OR #Steyer)" -bd 2019-12-19 -ed 2019-12-21 -o 20191219_21_candidates_tweets.json -l 50000")


if __name__ == '__main__':
    list_of_tweets = query_tweets("Trump OR Clinton", 10)

    #print the retrieved tweets to the screen:
    for tweet in query_tweets("Trump OR Clinton", 10):
        print(tweet)

    #Or save the retrieved tweets to file:
    file = open(“output.txt”,”w”)
    for tweet in query_tweets("Trump OR Clinton", 10):
        file.write(tweet.encode('utf-8'))
    file.close()





twitterscraper_text_part1='twitterscraper "(#democraticdebate OR #demdebate OR #debates2019 OR debate OR debates OR primary OR primaries OR election OR vote OR #debate OR #debates) AND '
twitterscraper_text_part2='(#candidate candidate)" -bd 2019-12-19 -ed 2019-12-20 -o 20191219_20_Klobuchar_tweets.json -l 20000'
text= ''

twitterscraper_text_full=(twitterscraper_text_part1, twitterscraper_text_part2, sep="")

twitterscraper_command=[]
for candidate in candidate_list:

my_str = my_str.replace(old[i],new[i])

list.append(text)
    
    
    for file in filelist:
    df2 = pd.read_json(file)
    state = file[8:-35]
    df2['State'] = state
    timestamp=df2['timestamp'][0]
    list.append(df2)
concat_jsons = pd.concat(list)



new = ['Amy','Klobuchar','r']
my_str = 'there are two much person a, person b, person c.'
old = ['a','b','c']

for i in range(len(old)):
    my_str = my_str.replace(old[i],new[i])

print(my_str)





#scrape twitter
#def scrapetwitter(topics, candidate, date):
#os.system #goes into terminal to scrape tweets
 #   twitterscraper("bernie sanders near:Boston" -bd 2016-02-01 -ed 2016-03-01 -o boston_bernie_20160201_0301.json -l 10000)


#this finds our jsons
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\Candidate_debate_date_raw\20191220")
path_to_json=(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\Candidate_debate_date_raw\20191220")
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)


test=pd.read_json(path_to_json+"\20191220_Klobuchar_tweets_test.json")

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






    


