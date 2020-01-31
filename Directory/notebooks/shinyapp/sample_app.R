
##testing

dftest <- subset(dataclean$norm_cos, dataclean$Candidate=="Biden")

leaflet(dataclean) %>% addTiles() %>%
  addCircles(lng = ~lng, lat = ~lat, weight = 1,
             radius =  ~(norm_cos*3000), popup=(norm_cos) 
  )

maptest <- leaflet(width = 400, height = 400)
maptest <- addTiles(maptest)
maptest <- setView(maptest, lng = -92,
                   lat = 35,
                   zoom = 3)%>%
  #addMarkers(lng=dataclean$lng[dataclean$State=="Wis"], lat=dataclean$lat[dataclean$State=="Wis"], 
  #addMarkers(lng=dataclean$lng, lat=dataclean$lat, 
  #          popup=paste("Tweets similarity score:"))%>%
  addCircles(lng = dataclean$lng[dataclean$State=="Wis"], lat = dataclean$lat[dataclean$State=="Wis"], 
             weight = 1,
             radius = (dataclean$norm_cos*3000),
             popup=paste("Tweets similarity score:",dataclean$norm_cos ))
r_colors <- rgb(t(col2rgb(colors()) / 255))
names(r_colors) <- colors()
maptest

barplot(y=subset(dataclean$norm_cos, dataclean$Candidate=="Biden"),x=dataclean$State)

#output$hist <- renderPlot({
# hist(rnorm(input$n))
#})





lng2=as.data.frame(unique(dataclean$lng))
lat2=as.data.frame(unique(dataclean$lat))

server <- function(input, output) {
  
  output$candidate_selected <- renderText({ 
    paste("Twitter engagement for", input$candidate, "for debate on",
          input$date, "Topics include", sep="\n")
  })
  output$map <- renderLeaflet({
    score <- as.data.frame(subset(dataclean$norm_cos, dataclean$Candidate==input$candidate))
    #score <-as.data.frame(subset(dataclean$norm_cos, dataclean$Candidate=="Biden"))
    df<-cbind(score, lng2, lat2)
    colnames(df)=c( "score", "lng", "lat")
    leaflet(df,width = 400, height = 400 ) %>% addTiles() %>%
      addCircles(lng = ~lng, lat = ~lat, weight = 1,
                 radius = ~score * 3000, popup = paste("Tweets similarity score:", 
                                                       data=(subset(dataclean$norm_cos, dataclean$Candidate==input$candidate)))
      )
    
  })
}




shinyApp(ui, server)



testscore <-as.data.frame(subset(dataclean$norm_cos, dataclean$Candidate=="Biden"))



leaflet(df,width = 400, height = 400 ) %>% addTiles() %>%
  addCircles(lng = ~lng, lat = ~lat, weight = 1,
             radius = ~score * 3000, popup = ~score
  )


library(reshape)
data_full$Candidate=as.character(gsub( "_tw", "", as.character(data_full$Candidate)))#one time fix
data_for_plot=melt(data_full, by=Candidate2)

(as.matrix(data_full))







