import pandas as pd
import numpy as np
import parse
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import spacy

torch.manual_seed(1)
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
words = []


def main():
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    num_states = len(states)

    devdata, numWords = parse.devParse()
    traindata, numTrain = parse.devParse(Train=True)

    for sentence in devdata:
        for word in sentence:
            words.append(str(word[0]).capitalize())
    words.append("UNKNOWN")

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, numWords+1, num_states)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)


    with torch.no_grad():
        print("PRE: ")
        input, whole = tokenizeWholeData(traindata)
        scores = model(input)
        print(scores)
        joblib.dump(words, 'phase2Words.pkl')
    #print(len(whole))

    count = 0
    #train the model
    for epoch in range(300):
        count += 1
        for i in range(len(traindata)):
            model.zero_grad()

            tokens = whole[i]
            sentence = traindata[i]

            #print(len(tokens))
            #print(len(sentence))

            sentence_in = torch.tensor(tokens, dtype=torch.long)
            targets = tokenizeTargets(sentence)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        joblib.dump(model, 'newRun\phase2test.pkl')
        if count == 50: joblib.dump(model, 'newRun\phase2n50.pkl')
        if count == 100: joblib.dump(model, 'newRun\phase2n100.pkl')
        if count == 150: joblib.dump(model, 'newRun\phase2n1500.pkl')
        if count == 200: joblib.dump(model, 'newRun\phase2n200.pkl')
        if count == 250: joblib.dump(model, 'newRun\phase2n250.pkl')
        print(count)

    joblib.dump(model, 'phase2test.pkl')

    with torch.no_grad():
        print("POST: ")
        input = tokenizeWholeData(traindata)
        scores = model(input)
        print(scores)

def tokenizeWholeData(data):
    tokens = []
    whole = []
    temp = []
    for sentence in data:
        temp = []
        for word in sentence:
            term = str(word[0]).capitalize()
            if term not in words:
                term = 'UNKNOWN'
            tokens.append(words.index(term))
            temp.append(words.index(term))
        whole.append(temp)
    
    return torch.tensor(tokens, dtype=torch.long), whole

def tokenize(data):
    #print(devdata)
    tokens = []

    for word in data:
            term = str(word[0]).capitalize()
            if term not in words:
                term = 'UNKNOWN'
            tokens.append(words.index(term))
    
    return torch.tensor(tokens, dtype=torch.long)
    nlp = spacy.load("en_core_web_lg")
    final = []
    string = ""
    for d in data:
        print(d)
        string += d[0]
        string += " "
    tokens = nlp(string)
    for token in tokens:
        print(token, token.i)
        final.append(token.i)
    print("sentences: ",  len(final))
    return torch.tensor(final, dtype=torch.long)

def tokenizeTargets(data):
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    targets = []
    for d in data:
        targets.append(states.index(d[1]))
    #print("targets: ", len(targets))
    return torch.tensor(targets, dtype=torch.long)



class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    

if(__name__ == "__main__") :
    main()