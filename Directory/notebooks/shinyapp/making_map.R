library(shiny)
library(leaflet)
library(maps)
library(ggplot2)
library(tmap)
require(dplyr)
library(sf)
library(raster)
library(dplyr)
library(spData)
library(tmaptools)
#library(mapdata)
#library(ggmap)
#library(spDataLarge)

tmap_tip()
#Basemaps can be disabled as follows, or by setting tmap_options(basemaps = NULL). 

data(World)
tm_basemap(NULL) +
  tm_shape(World)+
  tm_polygons("HPI")

data(state)
tm_basemap(NULL) +
  tm_shape(state)+
  tm_polygons("HPI")


states=map_data("state")
states2=data.frame(Sentiment_Score=c(1:49), Candidate=c(1:49))
statenames=(unique(states$region))
states2$States <- data.frame(matrix(unlist(statenames), nrow=length(statenames), byrow=T))
colnames(states2)=c("Sentiment_Score", "Candidate", "States")
data(state)









#get data
setwd("C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/data/cleaned")
tweets=read.csv("tweet_data_clean_newyork_Bernie Sanders _.csv")
tweets_boston=nrow(tweets[which(tweets$"State"=="boston"),])
tweets_newyork=nrow(tweets[which(tweets$"State"=="newyork"),])
tweets$lat[tweets$State=="boston"]=-71.058
tweets$lng[tweets$State=="boston"]=42.360




states=map_data("state")
states2=data.frame(Sentiment_Score=c(1:49), Candidate=c(1:49))
statenames=(unique(states$region))
states2$States <- data.frame(matrix(unlist(statenames), nrow=length(statenames), byrow=T))

data(state)

#Basemaps can be disabled as follows, or by setting tmap_options(basemaps = NULL). 

data(state)
tm_shape("state.abb") +
  tm_polygons("state.area") +
  tm_tiles("state.area") +
  tm_view(set.view = c(lon = 15, lat = 48, zoom = 4))


#myCorpus <- tm_map(myCorpus, removeWords,stopwords("english"))
myCorpus <- tm_map(myCorpus, stripWhitespace)





#EXAMPLE Semi-transparent tile layers can be added with tm_tiles. 
data(World)
tm_shape(World) +
  tm_polygons("life_exp") +
  tm_tiles("Stamen.TonerLines") +
  tm_view(set.view = c(lon = 15, lat = 48, zoom = 4))






