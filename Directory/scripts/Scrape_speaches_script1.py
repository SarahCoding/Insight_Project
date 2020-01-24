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
#specify url
quote_page="https://www.presidency.ucsb.edu/documents/remarks-des-moines-following-the-iowa-caucus-1"
quote_page2="https://www.presidency.ucsb.edu/documents/remarks-presidential-candidate-donald-trump-after-winning-primaries-montana-south-dakota"
#query the website and return html
page=urllib.request.urlopen(quote_page2)

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

date=soup.find('div', attrs=('class', 'field-docs-start-date-time'))
date=date.text
date=date.split("\n")[1]

text=soup.find('div', attrs=('class', 'field-docs-content'))
text=text.text

#Create a data frame of the soup outputs for future nlp
speechdata=pd.DataFrame(([[candidate_name, date, text]]))
speechdata.columns=['candidate', 'date', 'text']

df = pd.read_csv(r'speech_scraped_test.csv')

df2 = df.append(pd.DataFrame(data = speechdata), ignore_index=True, sort=True)

#this is not working?
df2.to_csv(r'speech_scraped_test2.csv')

