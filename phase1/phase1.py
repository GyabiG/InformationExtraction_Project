import sys
import pandas as pd
import numpy as np
import hmmlearn.hmm as hmm
import parse
import joblib

def main():
    #create and train HMM
    devdata = parse.devParse()
    #print(devdata)
    words = []
    for sentence in devdata:
        for word in sentence:
            words.append(str(word[0]).capitalize())
    words.append('UNKNOWN')

    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    listOfstates = []
    for sentence in devdata:
        currData = []
        for word in sentence:
            currData.append(states.index(word[1]))
        listOfstates.append(np.array(currData))


    simpleData = []
    for sentence in devdata:
        for word in sentence:
            simpleData.append(word)
    startProb = np.array([5/6,1/12,0,1/12,0])
    transitionProb = calcTransistion(states, simpleData)
    emmisionProb = calcEmmision(states, simpleData, words)

    num_states = len(states)
    model = hmm.CategoricalHMM(n_components=num_states, algorithm="viterbi", init_params="te")
    model.startprob_ = startProb
    model.transmat_ = transitionProb
    model.emissionprob_ = emmisionProb

    print(model.sample(3))
    print(words[3])
    print(words[22412])

    trainData = parse.devParse(Train=True)
    fitData = []
    for sentence in trainData:
        for word in sentence:
            term = str(word[0]).capitalize()
            if term not in words:
                term = 'UNKNOWN'
            fitData.append([words.index(term), states.index(word[1])])
    print(len(words))
    model.fit(fitData)

    joblib.dump(model, 'phase1test.pkl')
    joblib.dump(words, 'phase1Wordstest.pkl')

    #print(listOfstates)
    #model.fit(listOfstates)
    #print(model.sample(3))
    #print expectations
    inputWords = parse.inputFileParse()
    print("finding new")
    asNum = []
    lenghts = []
    for sentence in inputWords:
        currSentence = []
        for word in sentence:
            word = str(word).capitalize()
            if word not in words:
                word = 'UNKNOWN'
            currSentence.append([words.index(word)])
        X = model.predict(np.array(currSentence, dtype=int))
        print(X)
        count = 0
        collection = []
        tokenWord = []
        definition = []
        for word in X:
            #print(states[word])
            if states[word] == 'I-Term':
                tokenWord.append(sentence[count])
            if states[word] == 'I-Definition':
                definition.append(sentence[count])
            count += 1
        collection.append([tokenWord, definition])
        print(collection)
        break
        asNum.append(currSentence)
        print("Next Sentence")

    #asNum = np.array(asNum)
    #print(model.predict(asNum))
    
    """for x in asNum:
        x = x.reshape(1, -1)
        prediction = model.predict(x)
        print(prediction)"""

def calcEmmision(states, data, words):
    N = len(states)
    M = len(data) + 2

    B = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if words[j] == 'UNKNOWN':
                B[i,j] = 1
                break
            count = sum(1 for k in range(len(states)) if states[k] == states[i] and str(data[k][0]).capitalize() == words[j])
            total = sum(1 for k in range(len(states)) if states[k] == states[i])
            if total == 0 or count == 0: B[i,j] = 0
            else: B[i, j] = count / total
    #print(B)
    for i in range(N):
        total = 0
        for j  in range(M):
            total += B[i,j]
        if total != 0:
            for j in range(M):
                B[i,j] = B[i,j]/total
    return B

def calcTransistion(states, data):
    M = len(data)
    N = len(states)

    counts = np.zeros((N,N))
    lastState = None
    for i in range(M):
        if lastState != data[i][1] and lastState != None:
            counts[states.index(lastState),states.index(data[i][1])] += 1
        lastState = data[i][1]

    trans = np.zeros((N,N))
    for x in range(N):
        for y in range(N):
            sum = 0
            for k in range(N):
                sum += counts[x][k]
            if sum == 0: 
                for k in range(N):
                    trans[x][y] = 1/N
            else: trans[x][y] = counts[x][y]/sum
    return trans


if(__name__ == "__main__") :
    main()