#data visualization of tweets by topic and region - maps

library(leaflet)
library(maps)
library(shiny)


setwd("C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/data/cleaned")

tweets=read.csv("tweet_data_clean_newyork_Bernie Sanders _.csv")

tweets_boston=nrow(tweets[which(tweets$"State"=="boston"),])
tweets_newyork=nrow(tweets[which(tweets$"State"=="newyork"),])

tweets$lat[tweets$State=="boston"]=-71.058
tweets$lng[tweets$State=="boston"]=42.360

map <- leaflet(width = 400, height = 400)
map <- addTiles(map)
map <- setView(map, lng = -73,
               lat = 41,
               zoom = 6)

# Show the map:
map_filled <- map %>%
  addTiles() %>%  # Add default OpenStreetMap map tiles
  addMarkers(lng=-71.058, lat=42.360, popup=paste(tweets_boston,"Tweets mentioning 'Country'"))%>%
  addMarkers(lng=-73.935242, lat=40.730610, popup=paste(tweets_newyork,"Tweets mentioning 'Country'"))

map_filled # Print the map
