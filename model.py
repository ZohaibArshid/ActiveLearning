# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:53:01 2021

@author: Zobi Tanoli
"""
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import csv
import nltk
from nltk import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import statistics


data = pd.read_csv('itor5(smallmargin).csv', encoding='Latin1')
txt= data['Text']
#print(type(txt))
txt= txt.values.tolist()
senti = data['Sentiment']#.value_counts() #(how to counts the value of evrey class in data)
#print(sent)

#print(data['Sentiment'].value_counts())

Etype=[]
filter_words = []
f1score = []
fspos = []
fsneg = []
fsneu = []
recallscore = []
rspos=[]
rsneg = []
rsneu = []
precisionscore = []
pspos = []
psneg = []
psneu = []
accuracyscore = []

#Creating the class object
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

flat_ls = [item for sublist in filter_words for item in sublist] # make list of lists to single list
#print(flat_ls)
text = pd.Series(flat_ls) # convert list to Pandas series
#print(type(text))
#print(len(text))

Etype.append(flat_ls)
Etype.append(senti)

for x in range(50):
    X_train, X_test, y_train, y_test= train_test_split(Etype[0], Etype[1] ,test_size=0.2)
    #print(len(X_train),len(X_test),len(y_train),len(y_test))
    print(y_train.value_counts())
    print(y_test.value_counts())
    #print(len(y_test))
    '''
    def simple_split(Etype,y, length, split_mark=0.8):
        if split_mark > 0. and split_mark < 1.0:
            n= int(split_mark*length)
        else:
            n= int(split_mark)
        X_train = Etype[:n].copy()
        print( X_train)
        X_test = Etype[n:].copy()
        #print( X_test)
        y_train = y[:n].copy()
        print(y_train.value_counts())
        y_test = y[n:].copy()
        #print( y_test)
        return X_train,X_test,y_train,y_test
    '''
    
    #vectorizer = CountVectorizer()
    #X_train,X_test,y_train,y_test= simple_split(Etype[0],Etype[1],len(Etype[0]))
    #print(len(X_train),len(X_test),len(y_train),len(y_test))
    #print(y_train)

    vectorizer = TfidfVectorizer(stop_words= 'english')
    X_train = vectorizer.fit_transform(X_train)
    #print(X_train)
    X_test = vectorizer.transform(X_test)
    
    feature_names = vectorizer.get_feature_names()
    #print(feature_names)

    classifier= SVC(kernel='linear', C= 1).fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    
    # F1 Score-----------------------------------
    fs= f1_score(y_test, y_predict, average='macro')
    f1score.append(fs)
    f1_pos=f1_score(y_test, y_predict, labels=['positive'], average='macro')
    fspos.append(f1_pos)
    f1_neg=f1_score(y_test, y_predict, labels=['negative'], average='macro')
    fsneg.append(f1_neg)
    f1_neu=f1_score(y_test, y_predict, labels=['neutral'], average='macro')
    fsneu.append(f1_neu)
    
    # Recall Score-------------------------------
    rs= recall_score(y_test, y_predict, average='macro')
    recallscore.append(rs)
    rs_pos=recall_score(y_test, y_predict, labels=['positive'], average='macro')
    rspos.append(rs_pos)
    rs_neg=recall_score(y_test, y_predict, labels=['negative'], average='macro')
    rsneg.append(rs_neg)
    rs_neu=recall_score(y_test, y_predict, labels=['neutral'], average='macro')
    rsneu.append(rs_neu)
    
    # precision score----------------------------
    ps= precision_score(y_test, y_predict, average='macro')
    precisionscore.append(ps)
    ps_pos=precision_score(y_test, y_predict, labels=['positive'], average='macro')
    pspos.append(ps_pos)
    ps_neg=precision_score(y_test, y_predict, labels=['negative'], average='macro')
    psneg.append(ps_neg)
    ps_neu=precision_score(y_test, y_predict, labels=['neutral'], average='macro')
    psneu.append(ps_neu)
    
    # accuracy---------------------------------
    acc= accuracy_score(y_test, y_predict)
    accuracyscore.append(acc)

'''
print(f1score)
print(fspos)
print(fsneg)
print(fsneu)

print(recallscore)
print(rspos)
print(rsneg)
print(rsneu)

print(precisionscore)
print(pspos)
print(psneg)
print(psneu)

print(accuracyscore)
'''
'''
df = pd.DataFrame({'F1 Score':f1score,
                   'F1 Pos Score': fspos, 
                   'F1 Neg Score': fsneg, 
                   'F1 Neu Score': fsneu,
                   'Recall Score': recallscore,
                   'Recall Pos Score': rspos, 
                   'Recall Neg Score': rsneg, 
                   'Recall Neu Score': rsneu,
                   'Precision Score': precisionscore,
                   'Precision Pos Score': pspos, 
                   'Precision Neg Score': psneg, 
                   'Precision Neu Score': psneu,
                   'Acurracy' : accuracyscore
                   })
print(df)
'''

def allmeans(f1score,fspos,fsneg,fsneu,recallscore,rspos,rsneg,rsneu,precisionscore,pspos,psneg,psneu,accuracyscore):
    a = statistics.mean(f1score)
    print('F1 Score:', a)
    b = statistics.mean(fspos)
    print('F1 Positive:',b)
    c = statistics.mean(fsneg)
    print('F1 Negative:',c)
    d = statistics.mean(fsneu)
    print('F1 Neutral:',d)
    e = statistics.mean(recallscore)
    print('Recall:',e)
    f = statistics.mean(rspos)
    print('Recall Positive:',f)
    i = statistics.mean(rsneg)
    print('Recall negative:', i)
    j = statistics.mean(rsneu)
    print('Recall neutral:',j)
    k = statistics.mean(precisionscore)
    print('Precision:',k)
    l = statistics.mean(pspos)
    print('Precision Positive:', l)
    m = statistics.mean(psneg)
    print('Precision Negative:',m)
    n = statistics.mean(psneu)
    print('Precision Neutral:', n)
    o = statistics.mean(accuracyscore)
    print('Accuracy:',o)
    
    

allmeans(f1score,fspos,fsneg,fsneu,recallscore,rspos,rsneg,rsneu,precisionscore,pspos,psneg,psneu,accuracyscore)   




def allstandarddeviations(f1score,fspos,fsneg,fsneu,recallscore,rspos,rsneg,rsneu,precisionscore,pspos,psneg,psneu,accuracyscore):
    a1 = statistics.stdev(f1score)
    print('F1 Score:', a1)
    b1 = statistics.stdev(fspos)
    print('F1 Positive:',b1)
    c1 = statistics.stdev(fsneg)
    print('F1 Negative:',c1)
    d1 = statistics.stdev(fsneu)
    print('F1 Neutral:',d1)
    e1 = statistics.stdev(recallscore)
    print('Recall:',e1)
    f1 = statistics.stdev(rspos)
    print('Recall Positive:',f1)
    i1 = statistics.stdev(rsneg)
    print('Recall negative:', i1)
    j1 = statistics.stdev(rsneu)
    print('Recall neutral:',j1)
    k1 = statistics.stdev(precisionscore)
    print('Precision:',k1)
    l1 = statistics.stdev(pspos)
    print('Precision Positive:', l1)
    m1 = statistics.stdev(psneg)
    print('Precision Negative:',m1)
    n1 = statistics.stdev(psneu)
    print('Precision Neutral:', n1)
    o1 = statistics.stdev(accuracyscore)
    print('Accuracy:',o1)
    

allstandarddeviations(f1score,fspos,fsneg,fsneu,recallscore,rspos,rsneg,rsneu,precisionscore,pspos,psneg,psneu,accuracyscore)   
    



'''
#Parameter Tuning

param_grid = {'C': [0.1, 0.25, 0.5, 1,1.25, 1.5,1.75, 2,2.25, 2.5, 2.75, 3, 3.5, 4, 5, 10, 100, 1000, 10000]} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train, y_train)


# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
'''


##### --------------- Plot ----------------- ########

'''
import numpy as np
import matplotlib.pyplot as plt


f1_positive =np.array([0.737,0.724, 0.740, 0.756,0.838])
f1_negative =np.array([0.701,0.590, 0.585, 0.553,0.582])
f1_neutral = np.array([0.115,0.604, 0.550, 0.533,0.391])

y= ['1', '2', '3', '4', '5']

plt.plot( y, f1_positive,'g')
plt.plot( y, f1_negative, 'r')
plt.plot( y, f1_neutral,'b')


plt.xlabel('No of Iteration')
plt.ylabel('Precision')
plt.legend(['Positive', 'Negitive', 'Neutral'], loc=4)
plt.title('Least Confident')
plt.show()
'''

'''
# -------- Accuracy -------------


# -------- Least Confident (Precision) ----------
[0.737,0.724, 0.740, 0.756,0.838]
[0.701,0.590, 0.585, 0.553,0.582]
[0.115,0.604, 0.550, 0.533,0.391]

# -------- Small Margin (Precision) ----------
[0.737,0.705, 0.749, 0.757,0.751]
[0.701,0.629, 0.538, 0.524,0.524]
[0.115,0.511, 0.469, 0.448,0.455]

# -------- Small Margin (Recall) -----------------
[0.970,0.938, 0.889, 0.857,0.813]
[0.394,0.451, 0.551, 0.601,0.602]
[0.010,0.140, 0.206, 0.207,0.289]

# -------- Higest Entropy (Recall) ------------------
[0.970,0.935, 0.894, 0.861,0.820]
[0.394,0.447, 0.450, 0.552,0.568]
[0.010,0.239, 0.305, 0.288,0.306]

# ------- Least Confident (Recall) ------------------
[0.970,0.941, 0.898, 0.865,0.838]
[0.394,0.440, 0.558, 0.615,0.582]
[0.010,0.210, 0.276, 0.277,0.391]

# -------- Smallest Margin (F1 Score) ------------------
[0.837,0.804, 0.811, 0.802,0.780]
[0.497,0.520, 0.540, 0.555,0.557]
[0.019,0.213, 0.280, 0.277,0.350]
'''
'''

accuracy =np.array([0.728,0.681, 0.664, 0.646,0.625])


y= ['1', '2', '3', '4', '5']

plt.plot( y, accuracy,'g')

plt.xlabel('No of Iteration')
plt.ylabel('Accuracy')
plt.title('Small Margin')
plt.show()
'''
'''
#------ Accuracy (Least Confident) -----------
[0.728,0.691, 0.679, 0.665,0.650]

#------ Accuracy (Highest Entropy) -----------
[0.728,0.685, 0.654, 0.651,0.627]

#------ Accuracy (Highest Entropy) -----------
[0.728,0.681, 0.664, 0.646,0.625]
'''



