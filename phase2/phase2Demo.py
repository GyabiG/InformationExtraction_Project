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
from phase2 import LSTMTagger

def start():
    model =  joblib.load("phase2n100.pkl")
    words = joblib.load('phase2Words.pkl')
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    #words.append("UNKNOWN")

    english = []
    data = []
    for line in sys.stdin:
        line = line.split('\n')[0]
        line = re.findall(r"[\w']+|[.,!?;]", line)
        print(line)
        sentence = []
        for word in line:
            english.append(word)
            word = word.capitalize()
            if word not in words:
                word = 'UNKNOWN'
            data.append(words.index(word))
        #data.append(sentence)
        #print(data)
        data = torch.tensor(data, dtype=torch.long)
        X = model(data)
        #print(X)
        BIOs = []
        best = []
        for x in X.tolist():
            i = x.index(max(x))
            BIOs.append(states[i])
            best.append(i)
        print(BIOs)
        #print(best)
        if  not BIOs.__contains__("B-Term"):
            print("No Definition")
        count = 0
        while count in range(len(english)):
            term = []
            if states[best[count]] == "B-Term":
                j = count
                term = []
                term.append(english[j])
                j += 1
                while j < len(X)-1 and states[best[j]] == "I-Term":
                    term.append(english[j])
                    j += 1
                while j < len(X)-1 and states[best[j]] == "O":
                    j += 1
                term.append("-")
                while j < len(X) - 1 and (states[best[j]] == "B-Definition" or states[best[j]] == "I-Definition"):
                    term.append(english[j])
                    j += 1
                print(*term)
            count += 1
        data = []
        english = []


if(__name__ == "__main__") :
    start()