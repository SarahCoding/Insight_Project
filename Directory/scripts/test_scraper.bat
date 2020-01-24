twitterscraper "near:Iowa" -bd 2016-02-01 -ed 2016-06-07 -o Ark_tweets_2016_02_01-06_08_2K.json -l 2000
twitterscraper "near:California" -bd 2016-02-01 -ed 2016-06-07 -o Cal_tweets_2016_02_01-06_08_2K.json -l 2000

cd C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw\scraped\topic_tweets


twitterscraper "believe AND bernie sanders" -bd 2016-02-01 -ed 2016-02-07 -o tweets_believe_sanders_20160201_20160207.json -l 10000

twitterscraper "american AND bernie sanders" -bd 2016-02-01 -ed 2016-02-07 -o tweets_american_sanders_20160201_20160207.json -l 10000

twitterscraper "political AND bernie sanders" -bd 2016-02-01 -ed 2016-02-07 -o tweets_political_sanders_20160201_20160207.json -l 10000


twitterscraper "political AND bernie sanders near:Boston" -bd 2016-02-01 -ed 2016-02-07 -o test.json -l 10000


####

twitterscraper "bernie sanders AND Boston" -bd 2016-02-01 -ed 2016-03-01 -o boston_bernie_20160201_0301.json -l 10000

twitterscraper "bernie sanders AND New York" -bd 2016-02-01 -ed 2016-03-01 -o newyork_bernie_20160201_20160201_0301.json -l 10000

twitterscraper "bernie sanders near:Philadelphia" -bd 2016-02-01 -ed 2016-03-01 -o philadelphia_bernie_20160201_0301.json -l 10000

twitterscraper "bernie sanders near:Baltimore" -bd 2016-02-01 -ed 2016-03-01 -o baltimoire_bernie_20160201_0301.json -l 10000


#campaign
twitterscraper "campain AND bernie sanders AND New York" -bd 2016-02-01 -ed 2016-03-01 -o campain_newyork_bernie_20160201_20160201_0301.json -l 10000
