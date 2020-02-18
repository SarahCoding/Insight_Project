#Creating a braplot of the topics discussed in the first 6 debates in 2019
library(ggplot2)
library(plyr)

setwd("C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/models/topic_output")

data=as.data.frame(read.csv("Manual_first6_2019.csv"))

data$scaled=(data$score)/6

data_mean=ddply(data, .(subject), summarize, mean=mean(score))



ggplot(data=data_mean, aes(x=reorder(subject, mean),mean)) +
  geom_bar(stat="identity", fill="steelblue", position="identity")+
  xlab("Topic")+ ylab("Score")+
  theme_minimal()+
  theme(axis.text.y = element_text(color = "grey20", size = 20, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.x = element_text(color = "grey20", size = 15, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 15, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"))+
  coord_flip()
