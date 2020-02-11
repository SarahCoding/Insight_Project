# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:10:53 2020

@author: sarah
"""
#Scrape speaches

from lxml import html
import lxml
import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import os




#Collecting all the websites needed to build a corpus of debates
#specify url
website="https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-candidates-debates-1960-2016"
#query the website and return html
page=urllib.request.urlopen(website) #will used specified url
#parse the html using BS
soup=BeautifulSoup(page,'html.parser')
soup
# Creating a list of debate links for future corpus
links=[]
for link in soup.find('tbody').find_all('a'):
    links.append(link.get('href'))
links=links[1:33] #chose debates from present until 2015
#now parsing through the debate links, individually

#debate_website_test="https://www.presidency.ucsb.edu/documents/presidential-debate-the-university-nevada-las-vegas"
#debate_website_test2="https://www.presidency.ucsb.edu/documents/democratic-candidates-debate-brooklyn-new-york"


def debate_scraper(debate_website):
    #test=(debate_website)
    #return test  
    global debate_data
    page=urllib.request.urlopen(debate_website) #will used specified url
    soup_debate=BeautifulSoup(page,'html.parser')
    # Take out the <div> of name and get its value
    debate_main=soup_debate.find('div', attrs=('class', 'wrapper')).find('div', attrs=('class', 'field-docs-content'))
    date=soup_debate.find('div', attrs=('class', "field-docs-start-date-time")).text.split("\n")[1]
    debate_text=debate_main.text
    participants=debate_main.find('p').text.split("\n")[1:]
    
    #Create a data frame of the soup outputs for future nlp
    debate_data=pd.DataFrame(([[date, participants, debate_text]]))
    debate_data.columns=['date', 'participants', 'text']
    return debate_data


debate_corpus=pd.DataFrame()
for debate_website in links:
    debate_data=debate_scraper(debate_website)
    debate_corpus=pd.DataFrame(debate_corpus.append(debate_data))
    debate_corpus['text']=debate_corpus['text'].str.replace(',', '')
    debate_corpus['text']=debate_corpus['text'].str.replace('.', '')
    debate_corpus['date']=debate_corpus['date'].str.replace(',', '')


########      
#debate scraper from more recent democratic debates
#Collecting all the websites needed to build a corpus of debates
os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\raw")
latest_links=pd.read_csv("links_to_latest_debates.csv")[0:].values.tolist()
latest_links=pd.read_csv("links_to_latest_debates.csv").values.tolist()
# Function to convert   
# Python program to convert a list 
# to string using list comprehension   
links2=[]
for link_number in range(8):
    links2.append(latest_links[link_number][0])
    

#debate_website_test3="https://www.rev.com/blog/transcripts/democratic-debate-transcript-houston-september-12-2019"
def debate_scraper_latest(debate_website):
    #test=(debate_website)
    #return test  
    page2=urllib.request.urlopen(debate_website) #will used specified url
    soup_debate2=BeautifulSoup(page2,'html.parser')
    # Take out the <div> of name and get its value
    date=soup_debate2.find('div', attrs=('class', 'fl-rich-text')).text
    debate_text=soup_debate2.find('div', attrs=('class', 'fl-callout-text')).text
    participants=soup_debate2.find('div', attrs=('class', 'fl-module fl-module-rich-text fl-node-5e186cbbd29ea')).text
    #Create a data frame of the soup outputs for future nlp
    debate_data=pd.DataFrame(([[date, participants, debate_text]]))
    debate_data.columns=['date', 'participants', 'text']
    return debate_data

debate_corpus2=pd.DataFrame()
for debate_website in links2:
    debate_data2=debate_scraper_latest(debate_website)
    debate_corpus2=pd.DataFrame(debate_corpus2.append(debate_data2))

debate_corpus=pd.DataFrame(debate_corpus.append(debate_corpus2))

debate_facts=debate_corpus[['date','participants']]
debate_facts.to_csv("debates_facts.csv", index=False)  


os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
#debate_corpus.to_csv("2016-2015_debates_corpus.csv", index=False)  
   

    
    
    

  
    
