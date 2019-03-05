#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:23:56 2018

@author: david
"""

#import nltk
#from nltk.corpus import stopwords
stopWords = []
import pandas as pd
import re

def tfidf(string):
    bow=[]
    tmp1=[]
    for elem in string:
        tmp=elem.split()
        for el in tmp:
            if el.lower() not in stopWords:
                tmp1.append(el.lower())
        bow.append(tmp1)
        tmp1=[]

    wordSet = set(bow[0]).union(set(bow[1]))
    for elem in bow[2:]:
        wordSet = wordSet.union(set(elem))

    wordDict=[]
    for i in range(len(bow)):
        wordDict.append(dict.fromkeys(wordSet, 0))

    i=0
    for elem in bow:
        for word in elem:
            wordDict[i][word]+=1
        i=i+1

    import pandas as pd
    pd.DataFrame(wordDict)

    def computeTF(wordDict, bow):
        tfDict = {}
        bowCount = len(bow)
        for word, count in wordDict.items():
            tfDict[word] = count/float(bowCount)
        return tfDict

    tfBow=[]
    i=0
    for elem in bow:
        tfBow.append(computeTF(wordDict[i], elem))
        i=i+1

    def computeIDF(docList):
        import math
        idfDict = {}
        N = len(docList)

        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for doc in docList:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1

        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))

        return idfDict

    idfs = computeIDF(wordDict)

    def computeTFIDF(tfBow, idfs):
        tfidf = {}
        for word, val in tfBow.items():
            tfidf[word] = val*idfs[word]
        return tfidf

    tfidfBow=[]
    i=0
    for elem in tfBow:
        tfidfBow.append(computeTFIDF(tfBow[i], idfs))
        i=i+1

    #data=[dframe]

    #dfconcat=pd.concat(data).reset_index()
    # loading gc 
    import gc 
  
    # get the current collection  
    # thresholds as a tuple 
    print("Garbage collection thresholds:", gc.get_threshold()) 
    
    
    TFIDF=pd.DataFrame(tfidfBow)
    TFIDF['max_value'] = TFIDF.max(axis=1)
    TFIDF

    filtre=0.6
    filtre2=0.3
    mot_min=4
    lmot=[]
    lscore=[]
    for i in list(TFIDF.index):
        #print('###############################################################')
        max=TFIDF['max_value'].loc[i]
        tmpn=[]
        tmpa=[]
        for elem in TFIDF.columns:
            if TFIDF[elem].loc[i] > max*filtre and elem != 'max_value':
                tmpn.append(elem)
                tmpa.append(TFIDF[elem].loc[i])
                

        lmot.append(tmpn)
        lscore.append(tmpa)


    #dfconcat['mot']=lmot
    #dfconcat['score']=lscore
    #del dfconcat['index']
    
    
    return lmot,lscore



def tfidf_affine(dfconcat):
    
    lmot=list(dfconcat['mot'])
    lscore=list(dfconcat['score'])

    for i in range(len(lmot)):
        try:
            lscore[i]=lscore[i].replace('[','').replace(']','').replace("'","").replace(' ','').strip().split(',')
            lmot[i]=lmot[i].replace('[','').replace(']','').replace("'","").replace(' ','').strip().split(',')
        except:
            continue
            
    mot_filt=[]
    for i in range(len(lmot)):
        m=[]
        k=1
        try:
            a=lscore[i].copy()
            b=lmot[i].copy()
        except:
            a=lscore[i].replace('[','').replace(']','').replace("'","").replace(' ','').strip().split(',')
            b=lmot[i].replace('[','').replace(']','').replace("'","").replace(' ','').strip().split(',')
            
            
        #nombre de mot : k
        while k < 5 and len(a) > 0:
            #print(k,len(a))
            m.append(b[a.index(max(a))])
            b.remove(b[a.index(max(a))])
            a.remove(max(a))
            k=k+1
        mot_filt.append(m)
        
    s=[]
    for elem in dfconcat['score']:
        try:
            a=elem
        except:
            a=elem.replace('[','').replace(']','').replace("'","").replace(' ','').strip().split(',')
        a=list(map(float, a))
        #print(elem,round(sum(a)/len(a),2))
        s.append(round(sum(a)/len(a),2))
    dfconcat['mots']=mot_filt
    dfconcat['tfidf_score']=s
    dfconcat.to_csv('tmp/df_tfide.csv',sep=',',index=False)
    return dfconcat

