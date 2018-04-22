from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import evaluation
import normalization

from pprint import pprint

def saveClassifier(clf, name):
    joblib.dump(clf, name)

def loadClassifer(name):
    return joblib.load(name)


def start():

    newsTrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)

    docsTrain = [' '.join(normalization.normalizeRaw(doc)) for doc in newsTrain.data]
    labelsTrain = newsTrain.target
    labelNames = newsTrain.target_names

    #evaluation.plotCategories(labelsTrain, labelNames)


    vectorizer = CountVectorizer()
    vectorsTrain = vectorizer.fit_transform(docsTrain)

    multinomialClf = MultinomialNB(alpha=0.5)
    multinomialClf.fit(vectorsTrain, labelsTrain)

    saveClassifier(multinomialClf, 'savedMultinomialClf.clf')

    newsTest = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    docsTest = [' '.join(normalization.normalizeRaw(doc)) for doc in newsTest.data]
    labelsTest = newsTest.target
    vectorsTest = vectorizer.transform(docsTest)

    predictedIndexes = multinomialClf.predict(vectorsTest)
    # predictedProbs = multinomialClf.predict_proba(vectorsTest)
    pprint(multinomialClf.score(vectorsTest, labelsTest))

    pprint(predictedIndexes)


    evaluation.printClassificationReport(labelsTest, predictedIndexes, labelNames)
    evaluation.printTopWordsPerLabel(multinomialClf, vectorizer, labelNames)
    # evaluation.plotConfusionMatrixHeatmap(labelsTest, predictedIndexes, labelNames)




def gridSearch(clf, params):
    pass


def classify(normalized_text):
    pass






