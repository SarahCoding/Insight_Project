#Creating a barplot of each states topic score
library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)




setwd("C:/Users/sarah/Dropbox/Insight_fellowship/Project/Directory/notebooks/shinyapp/Debate_resonator_2")
#data pull and clean
tweets_Debate_20191220=read.csv("Tweets_Debate_20191220.csv")
missing=as.data.frame(read.csv("missing_cleaned_values_debate_20191220.csv")[1:3])
main_data=tweets_Debate_20191220[,c("State", "Candidate", "cos")]
data_full=rbind(missing, main_data)


data_full$State2[data_full$State=="Col"]="Colorado"
data_full$State2[data_full$State=="Flo"]="Florida"
data_full$State2[data_full$State=="Mic"]="Michigan"
data_full$State2[data_full$State=="Min"]="Minnesota"
data_full$State2[data_full$State=="Nev"]="Nevada"
data_full$State2[data_full$State=="New"]="New Hampshire"
data_full$State2[data_full$State=="Nor"]="North Carolina"
data_full$State2[data_full$State=="Ohi"]="Ohio"
data_full$State2[data_full$State=="Pen"]="Pennsylvania"
data_full$State2[data_full$State=="Vir"]="Virginia"
data_full$State2[data_full$State=="Wis"]="Wisconsin"


Candidate2=(gsub( "_tw", "", as.character(data_full$Candidate)))#one time fix
data_full$Candidate <- Candidate2 #one time fix

data_scores=as.data.frame(summarise_at(group_by(data_full,Candidate, State2),vars(cos),funs(mean(.,na.rm=TRUE))))
data_scores$norm_cos=signif(((data_scores$cos)-min(data_scores$cos))/(max(data_scores$cos)-min(data_scores$cos))*100, 2)





p=ggplot(data=data_scores, aes(x=State2, y=norm_cos, fill=State2)) +
  geom_bar(stat="identity", position=position_dodge())+
  ylab("Similarity Scores (%)")+
  xlab("State")+
  ggtitle("Gun Violence Tweets Post December 19, 2019 Debate Across 11 Swing States")+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()+
  theme(legend.position="none",
        axis.text.y = element_text(color = "grey20", size = 12, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.x = element_text(color = "grey20", size = 12, angle = 90, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 12, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 12, angle = 90, hjust = .5, vjust = .5, face = "plain"))
print(p)

