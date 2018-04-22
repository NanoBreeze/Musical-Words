from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from collections import Counter
from pprint import pprint



def plotCategories(labels, labelNamesTrain):
    labelCounter = Counter(labels)

    labelIndexes, counts = zip(*dict(labelCounter).items())
    #plt.figure(figsize=(15, 10))
    plt.xlabel("Category Index")
    plt.ylabel("# of Documents")
    plt.title('# of documents for each category in 20NewsGroup')

    plt.bar(labelIndexes, counts, align='center')
    plt.xticks(labelIndexes, [labelNamesTrain[index] for index in labelIndexes], rotation=90)

    plt.show()



def printClassificationReport(labelsTrue, labelsPredicted, labelNames):
    pprint(classification_report(labelsTrue, labelsPredicted, target_names=labelNames))



def plotConfusionMatrixHeatmap(labelsTrue, labelsPredicted, labelNames):

    # note that there is a 1:1 correspondence between label index and labelNames, so an index of 0 means that its label is labelNames[0]
    cm = confusion_matrix(labelsTrue, labelsPredicted)
    df = pd.DataFrame(cm, index=labelNames, columns=labelNames)

    #fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(df, annot=True, fmt='g')

    plt.show()


def printTopWordsPerLabel(clf, vectorizer, labelNames, n=15):
    for i, category in enumerate(labelNames):
        topIndexes = np.argsort(clf.coef_[i])[-1 * n:]
        featureNames = vectorizer.get_feature_names()
        pprint(category + ': ' + ', '.join([featureNames[index] for index in topIndexes]))
        # return [featureNames[index] for index in topIndexes]
