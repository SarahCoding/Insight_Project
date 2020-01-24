#Merge and clean raw voter turnout data
#January 16, 2020

library(plyr)
library(dplyr)
library(ggplot2)

#loading data to merge
rawpath="C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/data/raw"

filenames=list.files(path=rawpath, full.names=TRUE)
datalist = lapply(filenames, function(df){d=read.csv(file=df,header=T)})

headers = as.data.frame(read.csv(filenames[1], skip = 1, header = F, nrows = 1, as.is = T))
d_08_gen = as.data.frame(read.csv(filenames[1], skip = 3, header = F))
colnames(d_08_gen)= headers
colnames(d_08_gen)[1]="State"
colnames(d_08_gen)


d_08_gen=datalist[[1]]
d_08_pri=datalist[[2]]
d_10_gen=datalist[[3]]
d_12_gen=datalist[[4]]
d_12_pri=datalist[[5]]
d_14_gen=datalist[[6]]
d_16_gen=datalist[[7]]
d_16_pri=datalist[[8]]
d_18_gen=datalist[[9]]

#Trying to see whether "State" is correctly labeled
for (dataframe in 1:length(datalist)) {
  print(dataframe)
  print(colnames(datalist[[dataframe]][1])=="State")
}

#Clean up colnames
for (i in c(1, 3, 4, 6, 7, 9)){
  headers = read.csv(datalist[[i]], skip = 1, header = F, nrows = 1, as.is = T)
  name =  assign(paste("a", i, sep = ""), i)    
  df = read.csv(datalist[[i]], skip = 3, header = F)
  colnames(df)= headers
  colnames(df)[1]="State"
  colnames()
}

headers = read.csv(file, skip = 1, header = F, nrows = 1, as.is = T)
df = read.csv(file, skip = 3, header = F)
colnames(df)= headers
colnames(df)[1]="State"
colnames()




for (i in 1:length(datalist)) {
  print(i)
  if (as.data.framedatalist[[i]][[1]][[1]]=="State"){
    headers = read.csv(as.data.frame(datalist[[i]]), skip = 1, header = F, nrows = 1, as.is = T)
    df = read.csv(datalist[[i]], skip = 3, header = F)
    colnames(df)= headers
    colnames(df)[1]="State"
    colnames()
  }
}




name_df_func <- function(df) {
  
  df_name <- as.name(paste0("df", df))
  bquote(.(df_name) <- .(df_name)[ , 1:2])
  
}

headers = read.csv(file, skip = 1, header = F, nrows = 1, as.is = T)
df = read.csv(file, skip = 3, header = F)
colnames(df)= headers
colnames(df)[1]="State"
colnames()

grep("State", headers)


setwd(rawdatapath)
clean= function(data.entry){}
multmerge = function(mypath){
  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = lapply(filenames, function(x){read.csv(file=x,header=T, skip=3)})
  Reduce(function(x,y) {merge(x,y)}, datalist)
}

d=multmerge(mypath = rawdatapath)


datalist = lapply(filenames, function(x){read.csv(file=x,header=T)})
merge.data.frame(datalist, by.x = "State")


#Some plots
as.numeric(sub("%","",d_08_gen$VEPTotalBallotsCounted))/100

d_08_gen$VEPTotalBallotsCounted=as.character(d_08_gen$"VEP Total Ballots Counted")
d_08_gen$VEPTotalBallotsCounted=as.numeric(sub("%","",d_08_gen$VEPTotalBallotsCounted))
hist(d_08_gen$VEPTotalBallotsCounted)
d_08_gen$State=as.character(d_08_gen$State)

ggplot(data = d_08_gen, aes(x=State, y=VEPTotalBallotsCounted))+
  geom_point(cex=2, alpha=0.75)+
  theme(axis.text=element_text(size=15),
      axis.title=element_text(size=15))+
  labs( y="% Votes Counted ", x="States")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(panel.grid.major = element_line(colour = "gray95"), panel.background = element_blank())
  




