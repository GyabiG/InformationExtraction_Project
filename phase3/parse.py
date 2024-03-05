#parse gold data into simpler form
import sys
import os
import pandas as pd
import numpy as np
import specialList

"""Parses gold data for HMM"""
def simpleParse():
    #take inputS
    for line in sys.stdin:
        if not line.isspace():
            #reformat
            splitline = line.split()
            #output refomatted data
            term = splitline[0]
            token = splitline[4]
            print(term + " " + token)

def devParse(Train=False, Eval = False):
    #take inputS
    numWords = 0
    data = []
    directory = 'data/deft_files/dev'
    if Train:
        directory = 'data/deft_files/train'
    if Eval: 
        directory = 'data/test_files/labeled/subtask_3'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            currData = []
            with open(filepath, 'r',encoding="utf-8") as file:
                for line in file:
                    #reformat
                    if not line.isspace():
                        numWords += 1
                        splitline = line.split()
                        #output refomatted data
                        term = splitline[0]
                        token = splitline[4]
                        if token == 'O':
                            token = token
                        elif token.split('-')[1] == "Term":
                            token = token.split('-')[0] + '-' + "Term"
                        elif token.split('-')[1] == "Definition":
                            token = token.split('-')[0] + '-' + "Definition"
                        elif token.split('-')[1] == "Qualifier":
                            token = 'O'
                        elif token.split('-')[2] == "Term":
                            token = token.split('-')[0] + '-' + "Term"
                        elif token.split('-')[2] == "Definition":
                            token = token.split('-')[0] + '-' + "Definition"
                        elif token.split('-')[2] == "Qualifier":
                            token = 'O'
                        else:
                            token = 'O'
                        currData.append([term, token])
            data.append(currData)
    return data, numWords

"""Parses gold data to get list of terms"""
#does not work
def termParse():
    #take input
    line = input()
    #reformat
    line = line.split()
    #output refomatted data
    term = line[0]
    print(term)

def modelDataParse():
    data = []
    for line in sys.stdin:
        if not line.isspace():
            #reformat
            splitline = line.split()
            finalData = []
            #output refomatted data
            term = ''.join(str(ord(c)) for c in splitline[0])
            finalData.append(term)
            print(term)
            finalData.append(int(splitline[2]))
            finalData.append(int(splitline[3]))
            token = ''.join(str(ord(c)) for c in splitline[4])
            finalData.append(token)
            #if splitline[5][0] == 'T': finalData.append(int(splitline[5][1:4]))
            #else: finalData.append(0)
            #if splitline[6][0] == 'T': finalData.append(int(splitline[6][1:4]))
            #else: finalData.append(0)
            #type = ''.join(str(ord(c)) for c in splitline[0])
            #finalData.append(type)

            data.append(np.array(finalData, dtype=float))
    return np.array(data, dtype=float)

def inputFileParse():
    #take inputS
    data = []
    directory = 'data/test_files/unlabeled/subtask_1'

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r',encoding="utf-8") as file:
                currData = []
                for line in file:
                    for word in line.split():
                        currData.append(word)
            data.append(currData)
    return data

def evalParse():
    #take inputS
    data = []
    directory = 'data/test_files/labeled/subtask_3'

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r',encoding="utf-8") as file:
                currData = []
                for line in file:
                    if not line.isspace():
                        splitline = line.split()
                        currData.append(splitline[0])
                data.append(currData)
    return data

def evalParseOther():
    #take inputS
    data = []
    directory = 'data/test_files/labeled/subtask_3'
    temp = ''

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r',encoding="utf-8") as file:
                currData = []
                for line in file:
                    if not line.isspace():
                        splitline = line.split()
                        currData.append(splitline[0])
                        temp += splitline[0] + ' '
        data.append(temp)
        temp = ''

    #print(len(data))
    return data


def NGRAM_devParse(nsize, Train=False, Eval = False):
    seenTerm = False
    seenDef = False
    afterCount = 0
    nBefore = specialList.specialList(nsize)
    #take inputS
    data = []
    directory = 'data/deft_files/dev'
    if Train:
        directory = 'data/deft_files/train'
    if Eval: 
        directory = 'data/test_files/labeled/subtask_3'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            currData = []
            with open(filepath, 'r',encoding="utf-8") as file:
                for line in file:
                    #reformat
                    if not line.isspace():
                        splitline = line.split()
                        #output refomatted data
                        term = splitline[0]
                        token = splitline[4]
                        if token == 'O':
                            token = token
                        elif token.split('-')[1] == "Term":
                            seenTerm = True
                            afterCount = 0
                            token = token.split('-')[0] + '-' + "Term"
                        elif token.split('-')[1] == "Definition":
                            seenDef = True
                            afterCount = 0
                            token = token.split('-')[0] + '-' + "Definition"
                        elif token.split('-')[1] == "Qualifier":
                            token = 'O'
                        elif token.split('-')[2] == "Term":
                            token = token.split('-')[0] + '-' + "Term"
                        elif token.split('-')[2] == "Definition":
                            token = token.split('-')[0] + '-' + "Definition"
                        elif token.split('-')[2] == "Qualifier":
                            token = 'O'
                        else:
                            token = 'O'

                        
                        if seenDef or seenTerm:
                            currData.extend(nBefore.r())
                            nBefore.clear()
                            currData.append([term, token])
                            if token == 'O':
                                afterCount += 1
                                if afterCount >= nsize:
                                    data.append(currData)
                                    seenDef = False
                                    seenTerm = False
                                    afterCount = 0
                                    nBefore.clear()
                        else:
                            nBefore.add([term, token])  
    return data


