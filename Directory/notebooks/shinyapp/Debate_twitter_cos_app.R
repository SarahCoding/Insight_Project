library(shiny)
library(leaflet)
library(Rcpp)
library(maps)

library(dplyr)
library(tidyr)
  
setwd("C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/data/cleaned")
  
tweets_Debate_20191220=read.csv("Tweets_Debate_20191220.csv")
links=read.csv("C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/data/raw/links_to_latest_debates.csv")
missing=as.data.frame(read.csv("missing_cleaned_values_debate_20191220.csv")[1:3])

main_data=tweets_Debate_20191220[,c("State", "Candidate", "cos")]
data_full=rbind(missing, main_data)

dataclean=as.data.frame(summarise_at(group_by(data_full,Candidate, State),vars(cos),funs(mean(.,na.rm=TRUE))))
dataclean$norm_cos=signif(((dataclean$cos)-min(dataclean$cos))/(max(dataclean$cos)-min(dataclean$cos))*100, 2)
Candidate2=as.data.frame(gsub( "_tw", "", as.character(dataclean$Candidate)))#one time fix
dataclean$Candidate <- Candidate2 #one time fix




#Some candidates have 0 tweets with debate mention
#if nrow(as.data.frame(subset(dataclean$State, dataclean$Candidate=="x")))=0 for x in dataclean$Candidate:
#  then subset(dataclean$Candidate, datadataclean$State=="0"
#              alldata$Group2[alldata$Group2=="aHYA"]="Young"



#add geographical information for leaflet below
#Col Flo Mic Min Nev New Nor Ohi Pen Vir Wis 
dataclean$lng[dataclean$State=="Col"]=-105.358887
dataclean$lat[dataclean$State=="Col"]=39.113014
dataclean$lng[dataclean$State=="Flo"]=-81.760254
dataclean$lat[dataclean$State=="Flo"]=27.994402
dataclean$lng[dataclean$State=="Mic"]=-84.506836
dataclean$lat[dataclean$State=="Mic"]=44.182205
dataclean$lng[dataclean$State=="Min"]=-94.636230
dataclean$lat[dataclean$State=="Min"]=46.392410
dataclean$lng[dataclean$State=="Nev"]=-117.224121
dataclean$lat[dataclean$State=="Nev"]=39.876019
dataclean$lng[dataclean$State=="New"]=-71.500000
dataclean$lat[dataclean$State=="New"]=44.000000
dataclean$lng[dataclean$State=="Nor"]=-80.793457
dataclean$lat[dataclean$State=="Nor"]=35.782169
dataclean$lng[dataclean$State=="Ohi"]=-82.9071
dataclean$lat[dataclean$State=="Ohi"]=40.4173
dataclean$lng[dataclean$State=="Pen"]=-77.1945
dataclean$lat[dataclean$State=="Pen"]=41.2033
dataclean$lng[dataclean$State=="Vir"]=-78.6569
dataclean$lat[dataclean$State=="Vir"]=37.4316
dataclean$lng[dataclean$State=="Wis"]=-89.500000
dataclean$lat[dataclean$State=="Wis"]=44.500000

States=as.data.frame(unique(dataclean$State))


##webapp
ui = fluidPage(
  titlePanel("Debate Impact on Twitter"),
  sidebarLayout(
  sidebarPanel(
    helpText("Create maps of engagement scores for a candidate one week following a debate."),
    
    selectInput("candidate",
                label = "Choose a candidate to display",
                choices = c("Candidate",
                            "Joe Biden"="Biden", 
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


server <- function(input, output) {
  
  output$candidate_selected <- renderText({ 
    paste("Twitter engagement for", input$candidate, "for debate on",
          input$date, "Topics include", sep='\n')
  })
  output$map <- renderLeaflet({
    df <- subset(dataclean$norm_cos, dataclean$Candidate==input$candidate)
    leaflet(width = 400, height = 400) %>%
      addTiles()%>%
      setView(map, lng = -92,
              lat = 35,
              zoom = 4)%>%
      addCircles(lng=dataclean$lng, lat=dataclean$lat, weight=1, 
                 radius=df*3000,
                 popup=paste("Tweets similarity score:", data=df))
  })
}




shinyApp(ui, server)

  
#Col Flo Mic Min Nev New Nor Ohi Pen Vir Wis 
#167 289 456 409 284 372  48 321 176 214  20



