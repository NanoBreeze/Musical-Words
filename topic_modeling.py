from nltk.corpus import brown
from nltk.corpus import reuters
from gensim import corpora, models
import normalization
import logging
import util
import os


class BrownCorpus:

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __iter__(self):
        for fileid in brown.fileids():
            yield self.dictionary.doc2bow(normalization.normalizeWords(brown.words(fileid)))


class ReutersCorpus:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __iter__(self):
        for fileid in reuters.fileids():
            yield self.dictionary.doc2bow(normalization.normalizeWords(reuters.words(fileid)))



def saveDictionary(dictionary, fname):
    '''@:param normalized_docs is an iterable of iterables. Eg,
    normalized_doc = [brown.words(fileid) for fileid in brown.fileids()]
    dictionary = corpora.Dictionary(normalized_doc)
    '''

    if not os.path.exists(os.path.dirname(fname)):
        util.mkdir(os.path.dirname(fname))

    dictionary.save(fname)
    dictionary.save_as_text(fname + '.txt')


def loadDictionary(fname):
    dictionary = corpora.Dictionary.load(fname)
    return dictionary



def saveCorpus(corpus, fname):

    if not os.path.exists(os.path.dirname(fname)):
        util.mkdir(os.path.dirname(fname))

    corpora.MmCorpus.serialize(fname, corpus)


def loadCorpus(fname):
    corpus = corpora.MmCorpus(fname)
    return corpus

def saveModel(model, fname):
    if not os.path.exists(os.path.dirname(fname)):
        util.mkdir(os.path.dirname(fname))

    model.save(fname)


def loadModel(fname, type=""):
    model = None

    if type == 'LDA':
        model = models.LdaModel.load(fname)
    elif type == 'TFIDF':
        model = models.TfidfModel.load(fname)
    else:
        raise Exception('Invalid type fo loadModel')

    return model


def initModelPipeline(corpusName="reuters"):
    '''Create and save all the necessary models. Assumes we use brown.words(..)
    returns lda model
    '''

    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    base_path = os.path.join('./pretrained/topics/', corpusName)

    normalized_docs = [normalization.normalizeWords(reuters.words(fileid)) for  fileid in reuters.fileids()] \
                      if corpusName == 'reuters' \
                      else [normalization.normalizeWords(brown.words(fileid)) for  fileid in brown.fileids()]

    dictionary = corpora.Dictionary(normalized_docs)
    saveDictionary(dictionary, os.path.join(base_path, corpusName + 'Dict.dict'))

    corpus = ReutersCorpus(dictionary) if corpusName == 'reuters' else BrownCorpus(dictionary)
    saveCorpus(corpus, os.path.join(base_path, corpusName + '_bow_norm.mm'))

    tfidfModel = models.TfidfModel(corpus)
    saveModel(tfidfModel, os.path.join(base_path, corpusName + '_tfidf_model.tfidf'))

    tfidfCorpus = tfidfModel[corpus]
    saveCorpus(tfidfCorpus, os.path.join(base_path, corpusName + '_tfidf_corpus.mm'))

    ldaModel = models.LdaModel(tfidfCorpus, id2word=dictionary, num_topics=90, passes=20, iterations=400)
    saveModel(ldaModel, os.path.join(base_path, corpusName + '_lda_model.lda'))

    ldaCorpus = ldaModel[tfidfCorpus]
    saveCorpus(ldaCorpus, os.path.join(base_path, corpusName + '_lda_corpus.mm'))


def loadPretrainedLDACorpusAndModel(base_path='./pretrained/topics/reuters', corpusName='reuters'):
    corpus = loadCorpus(os.path.join(base_path, corpusName + '_lda_corpus.mm'))
    model = loadModel(os.path.join(base_path, corpusName + '_lda_model.lda', 'LDA'))

    return corpus, model

    # corpus, model = topic_modeling.loadPretrainedLDACorpusAndModel()

    # print(model.print_topics(10))

    # for doc in corpus:
    #     print(doc)
