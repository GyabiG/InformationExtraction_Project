import sys
import parse
import hmmlearn.hmm as hmm
import joblib

def main():
    model =  joblib.load("phase1Demo.pkl")
    words = joblib.load('phase1DemoWords.pkl')
    states = ['O','B-Definition', 'I-Definition','B-Term','I-Term']
    
    goldData = parse.devParse(Eval=True)
    testData = parse.evalParse()
    processedData = []

    for sentence in testData:
        sequence = []
        for word in sentence:
            word = word.capitalize()
            if word not in words:
                word = 'UNKNOWN'
            sequence.append([words.index(word)])
        X = model.predict(sequence)
        sequence = []
        count = 0
        for state in X:
            sequence.append([sentence[count], states[state]])
            count += 1
        processedData.append(sequence)

    labelMatch(goldData, processedData)
    partial(goldData,processedData,state)
    exactMatch(goldData, processedData, states)

def partial(parse1, parse2, states):
    defs = findDefs(parse1,parse2, states)
    testDefs = defs[1]
    goldDefs = defs[0]

    score = 0
    totalGold = 0
    totalGuess = 0
    missed = 0
    for def1 in testDefs:
        for def2 in goldDefs:
            if len(def1) == 0 and len(def2) != 0: missed += 1
            set1 = set(def2)
            set2 = set(def1)
            common_elements = set1.intersection(set2)
            score += len(common_elements)
            totalGold += len(def2)
            totalGuess += len(def1)
    recall = score/totalGold
    precision = score/totalGuess
    print("Partial: \n" + "  reacall: " + str(recall)
        + "\n precision: " + str(precision) + 
        "\nf-score: " + str((2*recall*precision)/(recall+precision)))

def exactMatch(goldData, testData, states):
    defs = findDefs(goldData,testData, states)
    testDefs = defs[1]
    goldDefs = defs[0]

    score = 0
    total = 0
    missed = 0
    totalGold = 0
    totalGuess = 0
    for def1 in testDefs:
        for def2 in goldDefs:
            if len(def1) == 0 and len(def2) != 0: missed += 1
            if def1 == def2:
                score += (len(def1) + len(def2))
            totalGold += len(def2)
            totalGuess += len(def1)

    recallP = recall(score,totalGold)
    precisionP = precision(score,totalGuess)
    print("Exact: \n" + "  reacall: " + str(recallP)
        + "\n precision: " + str(precisionP) + 
        "\nf-score: " + str((2*recallP*precisionP)/(recallP+precisionP)))

def findDefs(goldData, testData, states):
    goldDefs = []
    testDefs = []
    for index in range(len(goldData)):
        tempGold = goldData[index]
        term = []
        for count in range(len(tempGold)):
            if tempGold[count][1] == "B-Term":
                term.append(tempGold[count][0])
                while count < len(tempGold)-1 and tempGold[count+1][1] == "I-Term":
                    term.append(tempGold[count+1][0])
                    count += 1
                while count < len(tempGold)-1 and tempGold[count+1][1] == "O":
                    count += 1
                term.append(" ")
                while count < len(tempGold)-1 and (tempGold[count+1][1] == "B-Definition" or tempGold[count+1][1] == "I-Definition"):
                    term.append(tempGold[count+1][0])
                    count += 1
                goldDefs.append(term)
                term = []
    
    for i in range(len(testData)):
        tempTest = testData[i]
        term = []
        for j in range(len(tempTest)):
            if tempTest[j][1] == "B-Term":
                term.append(tempTest[j][0])
                while j < len(tempTest)-1 and tempTest[j+1][1] == "I-Term":
                    term.append(tempTest[j+1][0])
                    j += 1
                term.append("-")
                while j < len(tempTest)-1 and (tempTest[j+1][1] == "B-Definition" or tempTest[j+1][1] == "I-Definition"):
                    term.append(tempTest[j+1][0])
                    j += 1
                testDefs.append(term)
                term = []
    return[goldDefs, testDefs]


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
    return count/total

def precision(score, total):
    return score/total

def fscore(recall, precision):
    return (2*recall*precision)/(recall+precision)

if(__name__ == "__main__") :
    main()