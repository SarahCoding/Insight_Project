

#Debate_20191220 resonatory deploy
library(dplyr)
library(leaflet)
library(maps)
library(Rcpp)
library(rsconnect)
library(shiny)
library(tidyr)
library(reshape)
#rsconnect::setAccountInfo(name='sarahcoding', token='0FBD24670D53516D9A94DA892DEF8FBC', secret='zVjk5Bc2xNliBrOialxeh/XENbgIjTo8nVvwUOtM')
#rsconnect::deployApp('C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/notebooks/shinyapp/Debate_resonator/Debate_twitter_cos_app_withmissing.R')

setwd("C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/notebooks/shinyapp/Debate_resonator_2/")
#data pull and clean
tweets_Debate_20191220=read.csv("Tweets_Debate_20191220.csv")
missing=as.data.frame(read.csv("missing_cleaned_values_debate_20191220.csv")[1:3])
main_data=tweets_Debate_20191220[,c("State", "Candidate", "cos")]
data_full=rbind(missing, main_data)


Candidate2=(gsub( "_tw", "", as.character(data_full$Candidate)))#one time fix
data_full$Candidate <- Candidate2 #one time fix

#add geographical information for leaflet below
#Col Flo Mic Min Nev New Nor Ohi Pen Vir Wis 
data_full$lng[data_full$State=="Col"]=-105.358887
data_full$lat[data_full$State=="Col"]=39.113014
data_full$lng[data_full$State=="Flo"]=-81.760254
data_full$lat[data_full$State=="Flo"]=27.994402
data_full$lng[data_full$State=="Mic"]=-84.506836
data_full$lat[data_full$State=="Mic"]=44.182205
data_full$lng[data_full$State=="Min"]=-94.636230
data_full$lat[data_full$State=="Min"]=46.392410
data_full$lng[data_full$State=="Nev"]=-117.224121
data_full$lat[data_full$State=="Nev"]=39.876019
data_full$lng[data_full$State=="New"]=-71.500000
data_full$lat[data_full$State=="New"]=44.000000
data_full$lng[data_full$State=="Nor"]=-80.793457
data_full$lat[data_full$State=="Nor"]=35.782169
data_full$lng[data_full$State=="Ohi"]=-82.9071
data_full$lat[data_full$State=="Ohi"]=40.4173
data_full$lng[data_full$State=="Pen"]=-77.1945
data_full$lat[data_full$State=="Pen"]=41.2033
data_full$lng[data_full$State=="Vir"]=-78.6569
data_full$lat[data_full$State=="Vir"]=37.4316
data_full$lng[data_full$State=="Wis"]=-89.500000
data_full$lat[data_full$State=="Wis"]=44.500000



data_scores=as.data.frame(summarise_at(group_by(data_full,Candidate, State, lng, lat),vars(cos),funs(mean(.,na.rm=TRUE))))
data_scores$norm_cos=signif(((data_scores$cos)-min(data_scores$cos))/(max(data_scores$cos)-min(data_scores$cos))*100, 2)


as.data.frame(summarise_at(group_by(data_full,Candidate, State, lng, lat),vars(cos),funs(nrow(.,na.rm=TRUE))))

data_full$weight=as.numeric(rep("1", nrow(data_full)))
data_counts=as.data.frame(summarise_at(group_by(data_full,Candidate, State, lng, lat),vars(weight),funs(sum(.,na.rm=TRUE))))




#States=as.data.frame(unique(data_scores$State))
lng2=as.data.frame(unique(data_scores$lng))
lat2=as.data.frame(unique(data_scores$lat))



##webapp
ui <-fluidPage(
  titlePanel("Debate Reactor"),
  sidebarLayout(
    sidebarPanel(
      helpText("Create maps ranking twitter reaction by state following debates."),
      selectInput("candidate",
                  label = "Choose a candidate to display",
                  choices = c( "Joe Biden"="Biden", 
                               "Pete Buttigieg"="Buttigieg",
                               "Amy Klobuchar"="Klobuchar", 
                               "Bernie Sanders"="Sanders",
                               "Tom Steyer"="Steyer",
                               "Elizabeth Warren"="Warren",
                               "Andrew Yang"="Yang"),
                  selected = "Candidate"),
      dateInput("date", label = h3("Date of debate"), value = "2019-12-20")),
    hr()),
  mainPanel(
    textOutput("candidate_selected"),
    leafletOutput("map"),
    p()
  )
)





server <- function(input, output, session) {
  
  output$candidate_selected <- renderText({ 
    paste("Twitter engagement for", input$candidate, "for debate on",
          input$date, "Topics include RACISM and GUN VIOLENCE.")
  })
  output$map <- renderLeaflet({
    score <- rank(as.data.frame(subset(data_scores$norm_cos, data_scores$Candidate==input$candidate)))
    #score <- rank(as.data.frame(subset(data_scores$norm_cos, data_scores$Candidate=="Biden")))
    count <-as.data.frame(subset(data_counts$weight, data_counts$Candidate==input$candidate))
    #count <-as.data.frame(subset(data_counts$weight, data_counts$Candidate=="Biden"))
    df<-cbind(score,count, lng2, lat2)
    colnames(df)=c( "score","count", "lng", "lat")
    leaflet(df,width = 400, height = 400 ) %>% addTiles() %>%
      addCircles(lng = ~lng, lat = ~lat, weight = 1, color="Blue",
                 radius = ~score * 15000, popup = paste( "Debate resonated #",data=rank(-(subset(data_scores$norm_cos,
                                                                                                 data_scores$Candidate==input$candidate)), ties.method = "last"),
                                                         "among 11 swing states"), group="Twitter Similarity")%>%
      addCircles(lng = ~lng, lat = ~lat, weight = 1, color = "Red",
                 radius = ~count * 3000, popup = paste(data=subset(data_counts$weight, data_counts$Candidate==input$candidate), "Tweets tweeted"), group="Twitter Count")%>%
      addLayersControl(
        overlayGroups = c("Twitter Similarity", "Twitter Count"),
        options = layersControlOptions(collapsed = FALSE)
      )
    
  })
}

shinyApp(ui, server)
                                                                                 