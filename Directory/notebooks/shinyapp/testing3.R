library(shiny)
library(leaflet)
library(RColorBrewer)


#https://rstudio.github.io/leaflet/shiny.html
library(shiny)
library(leaflet)
library(RColorBrewer)


ui <- bootstrapPage(
  tags$style(type = "text/css", "html, body {width:100%;height:100%}"),
  leafletOutput("map", width = "100%", height = "100%"),
  absolutePanel(top = 10, right = 10,
                selectInput("candidate", "Candidates",c( "Joe Biden"="Biden", 
                                                          "Pete Buttigieg"="Buttigieg",
                                                          "Amy Klobuchar"="Klobuchar", 
                                                          "Bernie Sanders"="Sanders",
                                                          "Tom Steyer"="Steyer",
                                                          "Elizabeth Warren"="Warren",
                                                          "Andrew Yang"="Yang"))
                ),
                checkboxInput("legend", "Show legend", TRUE)
  )


server <- function(input, output, session) {
  
  
  output$map <- renderLeaflet({
    # Use leaflet() here, and only include aspects of the map that
    # won't need to change dynamically (at least, not unless the
    # entire map is being torn down and recreated).
    leaflet(data_full) %>% addTiles() %>%
      fitBounds(~min(lng), ~min(lat), ~max(lng), ~max(lat))
  })
  
  # Incremental changes to the map (in this case, replacing the
  # circles when a new color is chosen) should be performed in
  # an observer. Each independent set of things that can change
  # should be managed in its own observer.
  observe({
    leafletProxy("map", data = data_full) %>%
      clearShapes() %>%
      addCircles(radius = ~10^norm_cos/10, weight = 1, color = "#777777",
                 fillOpacity = 0.7, popup = ~paste(norm_cos)
      )
  })
}

shinyApp(ui, server)