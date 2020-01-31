# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 08:56:48 2020

@author: sarah
"""


#attempting to create a list of text based on the speaker    
    sep_list=[]
    for p_elem in debate_main.find_all('p'):
        sep_list.append(p_elem.text)
    
    
    
    
    D_lastname=participants[participants.index("(D)")-1]
    R_lastname=participants[participants.index("(R)")-1]
    participants_list.append([D_lastname, R_lastname])

participants_list=[]

    for participants_list in participants:
    participants_list.append(participants[participants.index("(D)")-1])
   
    
    

    participants
    
    
    
    text=soup.find('div', attrs=('class', 'field-docs-content'))
    text=text.text
    
    