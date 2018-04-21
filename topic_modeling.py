from nltk.corpus import brown
from nltk.corpus import reuters
from gensim import corpora, models
import normalization
import logging


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

    dictionary.save(fname)
    dictionary.save_as_text('Text_version_' + fname + '.txt')


def loadDictionary(fname):
    dictionary = corpora.Dictionary.load(fname)
    return dictionary



def saveCorpus(corpus, fname):
    corpora.MmCorpus.serialize(fname, corpus)


def loadCorpus(fname):
    corpus = corpora.MmCorpus(fname)
    return corpus

def saveModel(model, fname):
    model.save(fname)


def loadModel(fname, type=""):
    model = None

    if type == 'LDA':
        model = models.LdaModel.load(fname)
    elif type == 'TFIDF':
        model = models.TfidfModel.load(fname)
    else:
        raise Exception('Invalide type fo loadModel')

    return model


def initModelPipeline(corpusName="reuters"):
    '''Create and save all the necessary models. Assumes we use brown.words(..)
    returns lda model
    '''

    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    normalized_docs = [normalization.normalizeWords(reuters.words(fileid)) for  fileid in reuters.fileids()] \
                      if corpusName == 'reuters' \
                      else [normalization.normalizeWords(brown.words(fileid)) for  fileid in brown.fileids()]

    dictionary = corpora.Dictionary(normalized_docs)
    saveDictionary(dictionary, corpusName + 'Dict.dict')

    corpus = ReutersCorpus(dictionary) if corpusName == 'reuters' else BrownCorpus(dictionary)
    saveCorpus(corpus, corpusName + '_bow_norm.mm')

    tfidfModel = models.TfidfModel(corpus)
    saveModel(tfidfModel, corpusName + '_tfidf_model.tfidf')

    tfidfCorpus = tfidfModel[corpus]
    saveCorpus(tfidfCorpus, corpusName + '_tfidf_corpus.mm')

    ldaModel = models.LdaModel(tfidfCorpus, id2word=dictionary, num_topics=90, passes=20, iterations=400)
    saveModel(ldaModel, corpusName + '_lda_model.lda')

    ldaCorpus = ldaModel[tfidfCorpus]
    saveCorpus(ldaCorpus, corpusName + '_lda_corpus.mm')


def loadPretrainedLDACorpusAndModel(corpusName='reuters'):
    corpus = loadCorpus(corpusName + '_lda_corpus.mm')
    model = loadModel(corpusName + '_lda_model.lda', 'LDA')

    return corpus, model

    # corpus, model = topic_modeling.loadPretrainedLDACorpusAndModel()

    # print(model.print_topics(10))

    # for doc in corpus:
    #     print(doc)
