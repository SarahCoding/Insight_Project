# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:10:53 2020

@author: sarah
"""
#Scrape speaches

from lxml import html
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
   
#now parsing through the debate links, individually
debate_corpus=[]
participants_list=[]
debate_website="https://www.presidency.ucsb.edu/documents/presidential-debate-the-university-nevada-las-vegas"
debate_website2="https://www.presidency.ucsb.edu/documents/democratic-candidates-debate-brooklyn-new-york"

def debate_scraper(debate_website):  
    page=urllib.request.urlopen(debate_website2) #will used specified url
    soup_debate=BeautifulSoup(page,'html.parser')
    # Take out the <div> of name and get its value
    debate_main=soup_debate.find('div', attrs=('class', 'wrapper')).find('div', attrs=('class', 'field-docs-content'))
    date=soup_debate.find('div', attrs=('class', "field-docs-start-date-time")).text.split("\n")[1]
    debate_text=debate_main.text
    participants=soup_main.find('p').text.split("\n")
    
    sep_list=[]
    for p_elem in debate_main.find_all('p'):
        sep_list.append(p_elem.text)
        
        
    
    
    
    
    D_lastname=participants[participants.index("(D)")-1]
    R_lastname=participants[participants.index("(R)")-1]
    participants_list.append([D_lastname, R_lastname])

    for participants_list in participants:
    participants_list.append(participants[participants.index("(D)")-1])
   
    
    

    participants
    
    
    
    text=soup.find('div', attrs=('class', 'field-docs-content'))
    text=text.text
    
    
  
    
    #Create a data frame of the soup outputs for future nlp
    speechdata=pd.DataFrame(([[candidate_name, date, text]]))
    speechdata.columns=['candidate', 'date', 'text']
    
    os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
    speechdata.to_csv("speechdata_" + str(candidate_name) +"_"+ str(date) + ".csv")  
    filename=("speechdata_" + str(candidate_name) +"_"+ str(date) + ".csv")
   