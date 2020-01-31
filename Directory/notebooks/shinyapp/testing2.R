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
library(mapdata)
#library(ggmap)


#library(spDataLarge)

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







states2=states(group_by(states$region))

data("World")

tm_shape(World) +
  tm_polygons("HPI")

tmap_mode("plot")
data(tweets)
tm_shape(tweets) +
  tm_polygons("State", palette = "RdYlBu")

# Add fill and border layers to nz shape
tm_shape(us_states) +
  tm_fill() +
  tm_borders() 
map_usa = tm_shape(us_states) + tm_polygons()
class(map_usa)
#> [1] "tmap"
map_usa1 = map_usa + tm_raster(alpha = 0.7)

tmap_tip()
Semi-transparent tile layers can be added with tm_tiles. 

data(World)
tm_shape(World) +
  tm_polygons("life_exp") +
  tm_tiles("Stamen.TonerLines") +
  tm_view(set.view = c(lon = 15, lat = 48, zoom = 4))











map <- leaflet(width = 400, height = 400)
map <- addTiles(map)
map <- setView(map, lng = -73,
               lat = 41,
               zoom = 6)
r_colors <- rgb(t(col2rgb(colors()) / 255))
names(r_colors) <- colors()


ui = fluidPage(
  selectInput("state", "Choose a Candidate:",
              list(`Candidate` = list("Bernie Sanders"))
  ),
  dateInput("date", label = h3("Date of speech"), value = "2016-02-01"),
    hr(),
  fluidRow(column(3, verbatimTextOutput("value"))),
  textOutput("result"),
  leafletOutput("map"),
  p()
)


server <- function(input, output, session) {
  
  points <- eventReactive(input$recalc, {
    cbind(rnorm(40) * 2 + 13, rnorm(40) + 48)
  }, ignoreNULL = FALSE)
  
  output$map <- renderLeaflet({
    leaflet(width = 400, height = 400) %>%
      addTiles()%>%
      setView(map, lng = -73,
              lat = 41,
              zoom = 6)%>%
      addMarkers(lng=-71.058, lat=42.360, popup=paste(tweets_boston,"Tweets mentioning 'Country'"))%>%
      addMarkers(lng=-73.935242, lat=40.730610, popup=paste(tweets_newyork,"Tweets mentioning 'Country'"))
  })
}

shinyApp(ui, server)
