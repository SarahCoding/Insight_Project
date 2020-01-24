# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:10:53 2020

@author: sarah
"""
#Scrape speaches
from datetime import date
from lxml import html
import requests
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

#website="https://www.presidency.ucsb.edu/documents/remarks-des-moines-following-the-iowa-caucus-1"

def speech_scraper(website):  
    page=urllib.request.urlopen(website) #will used specified url
    #parse the html using BS
    soup=BeautifulSoup(page,'html.parser')
    soup
    # Take out the <div> of name and get its value
    candidate=soup.find('div', attrs=('class', 'field-docs-person'))
    candidate=candidate.text
    candidate1=candidate.split("\n\n\n \n\n")[1]
    candidate2=candidate.split("\n\n\n\n")[0]
    candidate3=candidate2.split("\n\n\n \n\n")[1]
    candidate_name=candidate3.split("\n\n\n")[0]
    
    date=soup.find('span', attrs=('class', 'date-display-single'))
    date=date.text
    
    text=soup.find('div', attrs=('class', 'field-docs-content'))
    text=text.text
    
    #Create a data frame of the soup outputs for future nlp
    speechdata=pd.DataFrame(([[candidate_name, date, text]]))
    speechdata.columns=['candidate', 'date', 'text']
    
    #os.chdir(r"C:\Users\sarah\Dropbox\Insight_fellowship\Project\Directory\data\cleaned")
    #speechdata.to_csv("speechdata_" + str(candidate_name) +"_"+ str(date) + ".csv")
    return ("speechdata_" + str(candidate_name) +"_"+ str(date) + ".csv")
    
 
