# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 18:39:25 2021

@author: Zobi Tanoli
"""
from numpy import argmax
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk
from nltk import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import heapq
from scipy.stats import entropy


df= pd.read_csv('itor4(highentropy).csv', encoding='Latin1')
#print(df)

Etype=[]

txt = df['Text']
senti =df['Sentiment']
filter_words = []

lemmat = WordNetLemmatizer()
#words to stem
#Stemming the words
for word in txt:
    #lemm = WordNetLemmatizer(stem)
    word_list = nltk.word_tokenize(word)
    lemm =[(lemmat.lemmatize(w)) for w in word_list]
    sent = [' '.join(lemm)] # join list words as a string
    filter_words.append(sent)
        #lemm= lemmat.lemmatize(w)
        #filter_words.append(lemm)
#print(filter_words)

X_train = [item for sublist in filter_words for item in sublist] # make list of lists to single list
#print(len(X_train))
y_train = pd.Series(senti) # convert list to Pandas series
#print(len(y_train))
#print(type(text))
#print(len(text))

#Etype.append(flat_ls)
#Etype.append(senti)

#print(str(Etype))
'''
def simple_split(Etype,y, length, split_mark=0.8):
    if split_mark > 0. and split_mark < 1.0:
        n= int(split_mark*length)
    else:
        n= int(split_mark)
    X_train = Etype[:n].copy()
    #print( X_train)
    #print(type(X_train))
    X_test = Etype[n:].copy()
    print( X_test)
    print(type(X_test))
    y_train = y[:n].copy()
    #print( y_train)
    #print(type(y_train))
    y_test = y[n:].copy()
    #print( y_test)
    #print(type(y_test))
    return X_train,X_test,y_train,y_test

#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(stop_words= 'english')
X_train,X_test,y_train,y_test= simple_split(Etype[0],Etype[1],len(Etype[0]))
#print(len(X_train),len(X_test),len(y_train),len(y_test))
'''

# ----------------------- Randomly select reviews from another file ---------------------

csvfile = pd.read_csv('filter.csv', encoding= 'Latin1')
#print(len(csvfile))
df = csvfile['Text']
smple= df.sample(300)
#print(smple)
testset= smple.values.tolist()
#print(testset)
#print(len(testset))

# -----------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words= 'english')
#print(X_train)
X_train = vectorizer.fit_transform(X_train)
#print(X_train)
X_test = vectorizer.transform(smple)
#print(X_test)

feature_names = vectorizer.get_feature_names()
#print(feature_names)

#print(type(X_train),type(X_test),type(y_train),type(y_test))
#print((X_train),(X_test),(y_train),(y_test))
#SVC----> support vector classifier
classifier= svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
#y_predict = classifier.predict(X_test)
#print(len(y_predict))
#print(X_test)
#print(classifier.score(X_test,y_test))
clp= classifier.predict_proba(X_test)
#pd.DataFrame(classifier.predict_proba(X_test), columns=classifier.classes_)
#print(clp)

################ --------- Techniques ---------- ################

problist= clp.tolist()
#print(problist)
maxvalue = []
minusvalue = []

# ------------------- Least Confident Prediction ---------------------------

'''
for i in problist:
    max_value = max(i)
    maxvalue.append(max_value)

for x in maxvalue: # to get 1- maximum value from Classes(positive, negitive, neutral) 
    values = 1- x
    minusvalue.append(values)
#print(len(minusvalue))
lst_tuple = list(zip(minusvalue, testset)) # List of tuple for manage the value and indexes
#print(lst_tuple)
lst_tuple.sort(reverse = True)
#print(lst_tuple)
 # sort list in decending order
#print(lst_tuple)
k = 100 # Get Top k elements from Records
res = lst_tuple[:k] # Get Top K elements from Records
#print((res))
#print(len(res))
#outputindex = [x[1] for x in res] # used for getting 2nd element from list of tuple
#print(outputindex)
#print(len(output))
#res = {indexes[i]: minusvalue[i] for i in range(len(minusvalue))}
#print(res)

outputSentences = [x[1] for x in res]

data = pd.DataFrame(outputSentences)
data.to_csv('leastCP.csv')
'''

########### ------------------------ Smallest Margin ------------------------ #############

'''
small_margin = []

for i in problist:
    largest_integers = heapq.nlargest(2, i) 
    largest_integer = largest_integers[0]  # 39
    #print(largest_integer)
    second_largest_integer = largest_integers[1] # 26
    #print(second_largest_integer)
    number = largest_integer - second_largest_integer
    small_margin.append(number)
#print(small_margin)


lst_tuple = list(zip(small_margin, testset)) # List of tuple for manage the value and indexes
#print(lst_tuple)
lst_tuple.sort()
#print(lst_tuple)
 # sort list in decending order
k = 100 # Get Top k elements from Records
res = lst_tuple[:k] # Get Top K elements from Records
#print(res)
outputSentences = [x[1] for x in res]
#print(outputSentences)
    
data = pd.DataFrame(outputSentences)
data.to_csv('smallmargin.csv')
'''

############### ---------------------- Highest Entropy ------------------- ##############

'''
high_entropy = []
for i in problist:
    enpy = entropy(i , base=2)
    high_entropy.append(enpy)

#print(high_entropy)
lst_tuple = list(zip(high_entropy, testset)) # List of tuple for manage the value and indexes
#print(lst_tuple)
lst_tuple.sort(reverse = True)
#print(lst_tuple)
 # sort list in decending order
#print(lst_tuple)
k = 100 # Get Top k elements from Records
res = lst_tuple[:k] # Get Top K elements from Records

#print(len(res))

outputSentences = [x[1] for x in res]
#print(outputSentences)

data = pd.DataFrame(outputSentences)
data.to_csv('highentropy.csv')
'''





#lab=classifier.predict(X_test)
#print(len(lab))
#print((lab))
'''
########## Choosing Right Confidence level ###############
nc=np.arange(.35,1,.03)
acc=np.empty(22)
i=0
for k in np.nditer(nc):
    conf_ind=df["max"]>k
    X_train1 = np.append(X_train,X_unl[conf_ind,:],axis=0)
    y_train1 = np.append(y_train,df.loc[conf_ind,['lab']])
    clf = svm.SVC(kernel='linear', probability=True,C=1).fit(X_train1, y_train1)
    acc[i]=  clf.score(X_test, y_test)
    i = i + 1
'''

# ---------------------- Code for Random Selection ------------------


'''
csvfile = pd.read_csv('itorcheck.csv')
smple= csvfile.sample(10)
print(len(csvfile))
print(len(smple))


fil = csvfile.values.tolist()
lis = smple.values.tolist()

filtered_list = [string for string in fil if string not in lis]

text = pd.Series(filtered_list)
text.to_csv('RemainingData.csv')
'''



######## ---------------- Drop selected Reviews using concat Function ------------- ########

'''
import pandas as pd

df= pd.read_csv('itrator.csv', encoding='Latin1')
print(len(df))
smple = pd.read_csv('smallmargin.csv', encoding='Latin1')
values = (pd.merge(df,smple, indicator=True, how='outer')
         .query('_merge=="left_only"')
         .drop('_merge', axis=1))

print(len(values))

values.to_csv('itrator.csv')
'''







