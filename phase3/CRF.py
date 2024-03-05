import pandas as pd
import numpy as np
import parse
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
from phase3 import LSTMTagger
import sklearn_crfsuite
from sklearn_crfsuite import metrics

nlp = spacy.load("en_core_web_lg")


def main(): 
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    devdata = parse.NGRAM_devParse(3)
    trainData = parse.NGRAM_devParse(3,Train=True)
    evalData, numWords = parse.devParse(Eval=True)

    #print(devdata)

    devdata.extend(trainData)

    processedData = []
    tokens = []
    
    print("start processing data")
    tdev = tokenizeWholeData(devdata)
    for sentence in tdev:
        #print(X)
        processedData.append(X2features(sentence))


    for Y in devdata:
        tokens.append(tokenizeTargets(Y))
    print("Data proccesssed")

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    #print(processedData)
    print("start learning")
    #print(tokens)
    crf.fit(processedData, tokens)

    joblib.dump(crf, 'T2phase3CRFn3.pkl')    
    print("DONE")

    print("Start Testing")

    X_test = []
    tEval = tokenizeWholeData(evalData)
    for sentence in tEval:
        #print(X)
        X_test.append(X2features(sentence))

    y_test = []
    for Y in evalData:
        y_test.append(tokenizeTargets(Y))

    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred,
                      labels = states, average='macro'))
    
    print(metrics.flat_classification_report(y_test, y_pred))
    
    '''y_pred = crf.predict(X_test)
    sorted_labels = sorted(
    states,
    key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels
    ))'''


def X2features(sentence):
    array = []
    tokens = nlp(sentence)
    #print(tokens)
    for word in tokens:
        features = {
            'bias': 1.0,
            'index': word.i,
            'pos': word.pos_,
            'vector': word.vector_norm,
            'lower': word.lower_,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        array.append(features)
    return array

def tokenizeTargets(data):
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    targets = []
    for d in data:
        targets.append(d[1])
    #print("targets: ", len(targets))
    return targets

def tokenize(data):
    final = []
    string = ""
    for d in data:
        '''if d[0] == "link]":
            d[0] = "link"
        if d[0] == "3-D":
            d[0] = "3"'''
        
        nd = nlp(d[0])
        
        #print(nd[0])
        string += str(nd[0])
        string += " "
    return string

def tokenizeWholeData(data):
    whole = []
    temp = []

    for sentence in data:
        temp = tokenize(sentence)
        whole.append(temp)
    return whole

if(__name__ == "__main__") :
    main()