import pandas as pd
import numpy as np
import parse
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
from phase3 import LSTMTagger, tokenize
import sklearn_crfsuite

def main(): 
    lstm = joblib.load("phase3n1000.pkl")
    trainData, numWords = parse.devParse(Train=True)

    processedData = [[]]
    tokens = [[]]
    
    print("start processing data")
    for sentence in trainData:
        sequence = torch.tensor(tokenize(sentence), dtype=torch.float32)
        X = lstm(sequence.long())
        #print(X)
        processedData[0].extend(X2features(X))
        Y = tokenizeTargets(sentence)
        #print(Y)
        tokens[0].extend(Y)
    print("Data proccesssed")

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, c2=0.1,
        all_possible_transitions=True
    )
    
    print("start learning")
    crf.fit(processedData, tokens)

    joblib.dump(crf, 'phase3n1000crf.pkl')    
    print("DONE")


def X2features(X):
    array = []
    for x in X:
        features = {
            'prob 1': x[0],
            'prob 2': x[1],
            'prob 3': x[2],
            'prob 4': x[3],
            'prob 5': x[4],
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

if(__name__ == "__main__") :
    main()
