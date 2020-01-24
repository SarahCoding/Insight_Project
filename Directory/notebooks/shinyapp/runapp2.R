library(shiny)
library(leaflet)
library(RColorBrewer)


map <- leaflet(width = 400, height = 400)
map <- addTiles(map)
map <- setView(map, lng = -73,
               lat = 41,
               zoom = 6)

ui <-pageWithSidebar(
  
  # Application title
  headerPanel("Twitter Speech Reaction"),
  
  # Sidebar with controls to select the variable to plot against mpg
  # and to specify whether outliers should be included
  sidebarPanel(
    selectInput("variable", "Variable:",
                list("Bernie Sanders" = "BS")),
  
  mainPanel()
))

server <- function(input, output) {
  
  # Reactive expression for the data subsetted to what the user selected
  filteredData <-  eventReactive(input$recalc, {
    cbind(rnorm(40) * 2 + 13, rnorm(40) + 48)
  }, ignoreNULL = FALSE)
  
  output$mymap <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%  # Add default OpenStreetMap map tiles
      addMarkers(lng=-73.935242, lat=40.730610, popup=paste(tweets_newyork,"Tweets mentioning 'campaign'"))
  })
}


shinyApp(ui, server)