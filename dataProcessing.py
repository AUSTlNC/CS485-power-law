import os
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def getResults(docText, docName):
    fullDocLength = len(docText)
    tokens = word_tokenize(docText)
    wordCounter = collections.Counter(tokens)
    distinctWordCount = len(wordCounter.keys())
    # print(wordCounter)
    print("number of total distinct words: ", distinctWordCount)
    inverseCounter = collections.Counter(list(wordCounter.values()))
    # for k, v in wordCounter.items():
    #     if v not in set(inverseCounter.keys()):
    #         inverseCounter[v] = 1
    #     else:
    #         inverseCounter[v] += 1
    inverseCounter = dict(sorted(inverseCounter.items()))
    fig1 = plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(list(inverseCounter.keys()), list(inverseCounter.values()), 'o')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Log Word Frequency")
    plt.ylabel("Log occurrence of distinct words with a particular frequency")
    plt.title("log-log Frequency Plot for " + docName)
    # plt.loglog(list(newInverseCounter.keys()), list(newInverseCounter.values()))
    plt.show()
    # fig1.savefig("fig1_" + docName + ".jpeg")

    fig2 = plt.figure(figsize=(8, 6), dpi=100)
    plt.hist(list(wordCounter.values()), bins=10, log=True)
    plt.xlabel("Log word frequency")
    plt.ylabel("Log occurrence of distinct words with a particular frequency")
    plt.title("Histogram for " + docName)
    plt.show()
    # fig2.savefig("fig2_" + docName + ".jpeg")

    x, y = list(inverseCounter.keys()), list(inverseCounter.values())
    print(1 + len(x) / sum(np.log(np.array(x))))

    logX, logY = np.log10(x), np.log10(y)
    fig3 = plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(logX, logY, alpha=0.6)
    a, b = np.polyfit(logX, logY, 1)
    # A = np.vstack([logX, np.ones(len(x))]).T
    # a, b = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, logY))
    print("a: ", a)
    print("b: ", b)
    # newX = sorted(x)
    # newY = np.poly1d(np.polyfit(x, y, 1))(newX)

    newY = []
    for i in range(len(x)):
        newY.append(b + logX[i] * a)

    plt.plot(logX, newY)
    plt.xlabel("Log word frequency")
    plt.ylabel("Log occurrence of distinct words with a particular frequency")
    plt.title("Scatter Plot and Fitted Line for " + docName)
    plt.show()
    fig3.savefig("fig3_" + docName + ".jpeg")

    df = pd.DataFrame({'frequency': list(inverseCounter.keys()), 'occurrence': list(inverseCounter.values())})
    df = df.sort_values(by='frequency', ascending=True, ignore_index=True)
    freq = list(df['frequency'])
    occur = list(df['occurrence'])
    ccdf = []
    for i in range(len(freq)):
        sumOcc = 0
        for j in range(i + 1, len(freq)):
            sumOcc += occur[j]
        if sumOcc != 0:
            ccdf.append(sumOcc)
        else:
            ccdf.append(1)
    print(freq)
    print(ccdf)
    logFreq, logCcdf = np.log10(freq), np.log10(ccdf)
    a2, b2 = np.polyfit(logFreq, logCcdf, 1)
    print("a2: ", a2)
    print("b2: ", b2)

    fitted = []
    for i in range(len(logFreq)):
        fitted.append(b2 + np.dot(logFreq[i], a2))
    print("PLOTTING")
    fig4 = plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(logFreq, logCcdf, alpha=0.6)
    plt.plot(logFreq, fitted)
    plt.xlabel("Log word frequency")
    plt.ylabel("Log occurrence of distinct words with a particular frequency")
    plt.title("CCDF Scatter Plot and Fitted Line for " + docName)
    plt.show()
    fig4.savefig("fig4_" + docName + ".jpeg")


f = open("dracula.txt", "r", encoding='utf-8', errors='ignore')
docTextDracula = f.read()
f.close()

f = open("les-miserables.txt", "r", encoding='utf-8', errors='ignore')
docTextLesMiserables = f.read()
f.close()

getResults(docTextDracula, "Dracula")
getResults(docTextLesMiserables, "Les-Miserables")
