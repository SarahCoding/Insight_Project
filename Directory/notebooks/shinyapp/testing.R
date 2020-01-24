library(shiny)
library(leaflet)

library(leaflet)
library(maps)


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
