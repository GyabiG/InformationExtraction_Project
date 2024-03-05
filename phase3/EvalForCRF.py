import pandas as pd
import numpy as np
import parse
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
import re
import sys
import spacy
from phase3 import LSTMTagger
from CRF import X2features
import sklearn_crfsuite


def main():
    model =  joblib.load('phase3CRF2.pkl')
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    
    goldData, numWords = parse.devParse(Eval=True)
    #print(len(goldData))
    testData = parse.evalParseOther()
    #print(len(testData))
    #print(testData)
    processedData = []

    tokenTestData = tokenizeWholeData(testData)
    for sentence in tokenTestData:
        #print(sentence)
        X = model.predict(X2features(sentence))
        #print(X)
        prediction = []
        #print(X)
        for x in X:
            i = (states.index(x[0]))
            prediction.append(i)
        sequence = []
        count = 0
        #print(prediction)
        for state in prediction:
            sequence.append([sentence[count], states[state]])
            count += 1
        #print(sequence)
        processedData.append(sequence)

    #print(len(processedData))
    labelMatch(goldData, processedData)
    partial(goldData,processedData, states)
    exactMatch(goldData, processedData, states)


def tokenizeWholeData(data):
    whole = []
    temp = []

    for sentence in data:
        temp = tokenize(sentence)
        whole.append(temp)
    return whole

def tokenize(data):
    nlp = spacy.load("en_core_web_lg")
    final = []
    string = ""
    for d in data:
        
        nd = nlp(d)
        
        #print(nd[0])
        string += str(nd[0])
    return string

def partial(parse1, parse2, states):
    defs = findDefs(parse1, parse2, states)
    defs = defs[1]
    test = defs[1]
    gold = defs[0]

    testDefs = set(test[0])
    goldDefs = set(gold[0])

    testTerm = set(test[1])
    goldTerm = set(gold[1])

    DefsCommon = testDefs.intersection(goldDefs)
    TermsCommon = testTerm.intersection(goldTerm)

    recallP = recall(len(DefsCommon),len(goldDefs))
    precisionP = precision(len(DefsCommon),len(testDefs))
    print("Partial Definitions: \n", "  reacall: ", recallP,
        "\n precision: ", precisionP, 
        "\nf-score: ", fscore(recallP, precisionP))
    
    recallP = recall(len(TermsCommon),len(goldTerm))
    precisionP = precision(len(TermsCommon),len(testTerm))
    print("Partial Terms: \n", "  reacall: ", recallP,
        "\n precision: ", precisionP, 
        "\nf-score: ", fscore(recallP, precisionP))

def exactMatch(goldData, testData, states):
    defs = findDefs(goldData,testData, states)
    defs = defs[0]
    test = defs[1]
    gold = defs[0]

    testDefs = set(test[0])
    goldDefs = set(gold[0])

    testTerm = set(test[1])
    goldTerm = set(gold[1])

    DefsCommon = testDefs.intersection(goldDefs)
    TermsCommon = testTerm.intersection(goldTerm)

    recallP = recall(len(DefsCommon),len(goldDefs))
    precisionP = precision(len(DefsCommon),len(testDefs))
    print("Exact Definitions: \n", "  reacall: ", recallP,
        "\n precision: ", precisionP, 
        "\nf-score: ", fscore(recallP, precisionP))
    
    recallP = recall(len(TermsCommon),len(goldTerm))
    precisionP = precision(len(TermsCommon),len(testTerm))
    print("Exact Terms: \n", "  reacall: ", recallP,
        "\n precision: ", precisionP, 
        "\nf-score: ", fscore(recallP, precisionP))

def findDefs(goldData, testData, states):
    goldTerms = []
    testTerms = []
    goldDefs = []
    testDefs = []
    goldTermsP = []
    testTermsP = []
    goldDefsP = []
    testDefsP = []
    for i in range(len(goldData)):
        tempTest = goldData[i]
        term = []
        defi = []
        for j in range(len(tempTest)):
            if tempTest[j][1] == "B-Term":
                term.append(tempTest[j][0])
                while j < len(tempTest)-1 and tempTest[j+1][1] == "I-Term":
                    term.append(tempTest[j+1][0])
                    j += 1
                goldTerms.append(" ".join(map(str, term)))
                goldTermsP.extend(term)
                term = []
            if tempTest[j][1] == "B-Definition":
                defi.append(tempTest[j][0])
                while j < len(tempTest)-1 and tempTest[j+1][1] == "I-Definition":
                    defi.append(tempTest[j+1][0])
                    j += 1
                goldDefs.append(" ".join(map(str,defi)))
                goldDefsP.extend(defi)
                defi = []
    
    for i in range(len(testData)):
        tempTest = testData[i]
        term = []
        defi = []
        for j in range(len(tempTest)):
            if tempTest[j][1] == "B-Term":
                term.append(tempTest[j][0])
                while j < len(tempTest)-1 and tempTest[j+1][1] == "I-Term":
                    term.append(tempTest[j+1][0])
                    j += 1
                testTerms.append(" ".join(map(str, term)))
                testTermsP.extend(term)
                term = []
            if tempTest[j][1] == "B-Definition":
                defi.append(tempTest[j][0])
                while j < len(tempTest)-1 and tempTest[j+1][1] == "I-Definition":
                    defi.append(tempTest[j+1][0])
                    j += 1
                testDefs.append(" ".join(map(str, defi)))
                testDefsP.extend(defi)
                defi = []
    #print(testDefs)
    return[[goldDefs, goldTerms], [testDefs, testTerms]],[[goldDefsP, goldTermsP], [testDefsP, testTermsP]]


def labelMatch(gold, test):
    scount = 0
    ocount = 0
    ototal = 0
    oguess = 0
    BtermCount = 0
    Btermtotal = 0
    BtermGuess = 0
    ItermCount = 0
    ItermTotal = 0
    ItermGuess = 0
    BDefCount = 0
    BDefTotal = 0
    BDefGuess = 0
    IDefCount = 0
    IDefTotal = 0
    IDefGuess = 0
    for sentence in test:
        sentence2 = gold[scount]
        wcount = 0
        for token in sentence:
            token = token[1]
            if wcount >= len(sentence2):
                break;
            token2 = sentence2[wcount]
            token2 = token2[1]
            match token:
                case "O":
                    if token == token2:
                            ocount += 1
                    ototal += 1
                case "I-Definition":
                    if token == token2:
                            IDefCount += 1
                    IDefTotal += 1
                case "B-Definition":
                    if token == token2:
                            BDefCount += 1
                    BDefTotal += 1
                case "I-Term":
                    if token == token2:
                            ItermCount += 1
                    ItermTotal += 1
                case "B-Term":
                    if token == token2:
                            BtermCount += 1
                    Btermtotal +=1
            match token2:
                case "O":
                    oguess += 1
                case "I-Definition":
                    IDefGuess += 1
                case "B-Definition":
                    BDefGuess += 1
                case "I-Term":
                    ItermGuess += 1
                case "B-Term":
                    BtermGuess += 1
            wcount += 1
        scount += 1
    totalLabels = Btermtotal + ItermTotal + BDefTotal + ItermTotal + ototal


    oRecall = recall(ocount, ototal)
    oPrescision = precision(ocount, oguess)

    print("Tokens:")
    print("O: \n" + "  reacall: " + str(oRecall)
            + "\n precision: " + str(oPrescision)
            + "\nf-score: " + str(fscore(oRecall, oPrescision)))
    
    ItermRecall = recall(ItermCount, ItermTotal)
    ItermPrecision = precision(ItermCount, ItermGuess)
    print("I-Term: \n" + "  reacall: " + str(ItermRecall)
            + "\n precision: " + str(ItermPrecision)
            + "\nf-score: " + str(fscore(ItermRecall, ItermPrecision)))
    
    BtermRecall = recall(BtermCount, Btermtotal)
    BtermPrecision = precision(BtermCount, BtermGuess)
    print("B-Term: \n" + "  reacall: " + str(BtermRecall)
            + "\n precision: " + str(BtermPrecision)
            + "\nf-score: " + str(fscore(BtermRecall, BtermPrecision)))
    
    BDefRecall = recall(BDefCount, BDefTotal)
    BDefPrecision = precision(BDefCount, BDefGuess)
    print("B-Definition: \n" + "  reacall: " + str(BDefRecall)
            + "\n precision: " + str(BDefPrecision)
            + "\nf-score: " + str(fscore(BDefRecall, BDefPrecision)))
    
    IDefRecall = recall(IDefCount, IDefTotal)
    IDefPrecision = precision(IDefCount, IDefGuess)
    print("I-Definition: \n" + "  reacall: " + str(IDefRecall)
            + "\n precision: " + str(IDefPrecision)
            + "\nf-score: " + str(fscore(IDefRecall, IDefPrecision)))

def recall(count, total):
    if total == 0: return 0
    return count/total

def precision(score, total):
    if total == 0: return 0
    return score/total

def fscore(recall, precision):
    if recall+precision == 0: return 0
    return (2*recall*precision)/(recall+precision)

if(__name__ == "__main__") :
    main()