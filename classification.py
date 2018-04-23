from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import  Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import evaluation
import normalization
import os
import util

from pprint import pprint

def saveObj(clf, name):

    if not os.path.exists(os.path.dirname(name)):
        util.mkdir(os.path.dirname(name))

    joblib.dump(clf, name)

def loadObj(name):
    return joblib.load(name)


'''
def start():

    newsTrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)

    docsTrain = [' '.join(normalization.normalizeRaw(doc)) for doc in newsTrain.data]
    labelsTrain = newsTrain.target
    labelNames = newsTrain.target_names

    #evaluation.plotCategories(labelsTrain, labelNames)


    vectorizer = TfidfVectorizer()
    vectorsTrain = vectorizer.fit_transform(docsTrain)

    multinomialClf = MultinomialNB(alpha=0.5)
    multinomialClf.fit(vectorsTrain, labelsTrain)

    # saveClassifier(multinomialClf, 'savedMultinomialClf.clf')

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
'''


def trainClassifier(clfType='bayes', save=True):
    '''Trains MultinomialNB and SGDClassifier. Saves both of them to a file'''

    newsTrain = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)

    docsTrain = [' '.join(normalization.normalizeRaw(doc)) for doc in newsTrain.data]
    labelsTrain = newsTrain.target

    vectorizer = TfidfVectorizer()
    vectorsTrain = vectorizer.fit_transform(docsTrain)

    if clfType == 'bayes':
        hyperparams = {'alpha': np.linspace(0, 1, 20)}
        clf = gridSearch(MultinomialNB(), hyperparams, vectorsTrain, labelsTrain)
    else:
        hyperparams = {'alpha': [0.00001, 0.0005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
                      'epsilon': np.linspace(1, 0.1, 9),
                      'penalty' : ['l2', 'l1', 'elasticnet']
                      }
        clf = gridSearch(SGDClassifier(n_jobs=-1), hyperparams, vectorsTrain, labelsTrain)


    if save:
        saveObj(clf, os.path.join('./pretrained/classifiers', clfType + '.clf'))
        saveObj(vectorizer, os.path.join('./pretrained/vectorizers', clfType + '.vect'))

    return clf, vectorizer


def testClassifer(clf, vectorizer):
    newsTest = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    docsTest = [' '.join(normalization.normalizeRaw(doc)) for doc in newsTest.data]
    labelsTest = newsTest.target
    labelNames = newsTest.target_names
    vectorsTest = vectorizer.transform(docsTest)

    predictedLabels = clf.predict(vectorsTest)

    pprint("The parameters of the classifier are: " + str(clf.get_params()))
    pprint('Score: ' + str(clf.score(vectorsTest, labelsTest)))
    evaluation.printClassificationReport(labelsTest, predictedLabels, labelNames)
    evaluation.printTopWordsPerLabel(clf, vectorizer, labelNames)
    evaluation.plotConfusionMatrixHeatmap(labelsTest, predictedLabels, labelNames)


def gridSearch(clf, params, vectorsTrain, labelsTrain):
    """Applies gridsearch to find the best parameters. Then returns the model (wrapped within its gridsearch
    with those params automatically refitted"""

    grid_clf = GridSearchCV(clf, param_grid=params, cv=5)
    grid_clf.fit(vectorsTrain, labelsTrain)

    return grid_clf.best_estimator_


def classify(clf, vectorizer, normalized_text):
    """Returns the index of the text type, along with the probability of its correctness"""

    vectorsTest = vectorizer.transform([normalized_text])

    predictedLabel = clf.predict(vectorsTest)
    predictedProb = clf.predict_proba(vectorsTest)

    return predictedLabel, predictedProb









