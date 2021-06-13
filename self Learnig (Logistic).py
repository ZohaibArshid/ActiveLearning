# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:45:49 2021

@author: Zobi Tanoli
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

df= pd.read_csv('label-unlabel.csv', encoding='Latin1')
#6print(df)

Etype=[]

lst = df['Text'].values.tolist()
sent=df['Sentiment']
Etype.append(lst)
Etype.append(sent)

#print(str(Etype))

def simple_split(Etype,y, length, split_mark=0.8):
    if split_mark > 0. and split_mark < 1.0:
        n= int(split_mark*length)
    else:
        n= int(split_mark)
    X_train = Etype[:n].copy()
    #print( X_train)
    X_test = Etype[n:].copy()
    #print( X_test)
    y_train = y[:n].copy()
    #print( y_train)
    y_test = y[n:].copy()
    #print( y_test)
    return X_train,X_test,y_train,y_test

vectorizer = CountVectorizer()
X_train,X_test,y_train,y_test= simple_split(Etype[0],Etype[1],len(Etype[0]))
print(len(X_train),len(X_test),len(y_train),len(y_test))

#print(X_train)
X_train = vectorizer.fit_transform(X_train)
#print(X_train)
X_test = vectorizer.transform(X_test)
#print(X_test)

feature_names = vectorizer.get_feature_names()
#print(feature_names)

#print(type(X_train),type(X_test),type(y_train),type(y_test))
#print((X_train),(X_test),(y_train),(y_test))

classifier= LogisticRegression().fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(len(y_predict))
#print(X_test)
#print(classifier.score(X_test,y_test))
clp= classifier.predict_proba(X_test)
#pd.DataFrame(classifier.predict_proba(X_test), columns=classifier.classes_)
#print(clp)

################ --- Highest Predicted Probability Class Technique --- ################

problist= clp.tolist()
indexes=[]
for i in problist:
    max_value = max(i)
    maxindex= i.index(max_value)
    #print(max_value)
    indexes.append(maxindex)

senti=[]

for i in indexes:
    if i == 0:
        senti.append('negative')
    elif i == 1:
        senti.append('neutral')
    elif i == 2:
        senti.append('positive')

#print(senti)


data= pd.DataFrame(senti)
print(data.value_counts())
data.to_csv('itor.csv') 




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







'''
import pandas as pd
df= pd.read_csv('itrator.csv', encoding='Latin1')


print('SVM RESULT')
print(df['SVM'].value_counts())
print('Logistic RESULT')
print(df['Logistic'].value_counts())
print('Spacy RESULT')
print(df['Spacy'].value_counts())
'''














