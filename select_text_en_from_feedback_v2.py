# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:44:52 2021

@author: Zobi Tanoli
"""


import nltk
import numpy as np
import pandas as pd
import re
import itertools

'''
df= pd.read_csv('select_text_en_from_feedback_v2.csv', encoding='Latin1')
#print(df)
#df.info()
txt= df['Text'].values.tolist()
len(txt)
value=[]
for i in txt:
    #print(type(i))
    sp =str(i).split(' ')
    if len(sp) >= 3 :
        i=(' ').join(sp)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', i)
        value.append(sentences)
#print(value)

lisvalue= list(itertools.chain.from_iterable(value))  
print(len(lisvalue))

data= pd.DataFrame(lisvalue)
data.to_csv('itrator.csv') 
'''


df= pd.read_csv('filter.csv', encoding='Latin1')
#print(df)
#df.info()
txt= df['Text'].values.tolist()
#len(txt)
value=[]
for i in txt:
    #print(type(i))
    sp =str(i).split(' ')
    if len(sp) >= 6 :
        i=(' ').join(sp)
        value.append(i)
print(len(value))
dat= pd.DataFrame(value)
data = dat.dropna()
print(len(data))
data.to_csv('itrator.csv') 





