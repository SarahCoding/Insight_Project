#Merge and clean raw voter turnout data
#January 16, 2020

library(plyr)

#loading data to merge
rawdatapath="C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/data/raw"
d1=read.csv("2008 November General Election - Turnout Rates.csv")

setwd(rawdatapath)
multmerge = function(mypath){
  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = lapply(filenames, function(x){read.csv(file=x,header=T)})
  Reduce(function(x,y) {merge("State",y)}, datalist)
}
d=multmerge(mypath = rawdatapath)

filenames=list.files(path=mypath, full.names=TRUE)

datalist = lapply(filenames, function(x){read.csv(file=x,header=T)})

