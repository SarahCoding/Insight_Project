# Political Reactor

How far does your message go?
Link: sarahcoding.shinyapps.io/debate_reactor

This repository contains the codebase for Political Reactor: how far does your message go?. Political Reactor displays measures of social media reaction to the latest political event and maps the distribution of this reaction across the United States. 

## Getting Started


These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python - numpy, pandas, matplotlib, tf-idf, gensim, nltk, scikit-learn, os, json, glob, pyLDAvis, logging, beautifulsoup
R - ggplot, cran, maps, RShiny, rsconnect, car, stringr, multcomp, stats

### Break down into end to end tests

I chose an LDA model to topic analyse the debates because of higher coherence scores obtained when compared to a more stringent TF-IDF LDA model.

A cos similarity was the similarity score used to estimate how similar tweets were to topics discussed from debates because it deals better with semantics and shorter text compared to other methods (i.e. Jaccard)
```
```

### And coding style tests

Python
Scraping from debate websites
Scraping from twitter
Debate topic modelling
Twitter similarity scores

R:
RShiny webapp
```
```

## Deployment

This app runs on an RShiny server.


## Authors

* **Sarah Atwi** - *Initial work* 


## Acknowledgments

* Hat tip to twitterscraper, lda-modelling on gensim
* Inspired by collaboration and discussions at Insight Data Science, Toronto.

