import sys
import hmmlearn.hmm as hmm
import joblib
import re

def start():
    model =  joblib.load("phase1Demo.pkl")
    words = joblib.load('phase1DemoWords.pkl')
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']

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
            sentence.append(words.index(word))
        data.append(sentence)
        #print(data)
        X = model.predict(data)
        #print(X)
        BIOs = []
        for x in X:
            BIOs.append(states[x])
        print(BIOs)
        count = 0
        while count in range(len(english)):
            term = ["No Definition"]
            if states[X[count]] == "B-Term":
                j = count
                term = []
                term.append(english[j])
                j += 1
                while j < len(X)-1 and states[X[j]] == "I-Term":
                    term.append(english[j])
                    j += 1
                while j < len(X)-1 and states[X[j]] == "O":
                    j += 1
                term.append("-")
                while j < len(X) - 1 and (states[X[j]] == "B-Definition" or states[X[j]] == "I-Definition"):
                    term.append(english[j])
                    j += 1
            print(*term)
            count += 1
        data = []
        english = []


if(__name__ == "__main__") :
    start()