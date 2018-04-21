from nltk.corpus import brown
from gensim import corpora, models
import normalization
import logging


class BrownCorpus:

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __iter__(self):
        for fileid in brown.fileids():
            yield self.dictionary.doc2bow(normalization.normalizeWords(brown.words(fileid)))



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


def initModelPipeline():
    '''Create and save all the necessary models. Assumes we use brown.words(..)
    returns lda model
    '''

    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    normalized_docs = [normalization.normalizeWords(brown.words(fileid)) for  fileid in brown.fileids()]
    dictionary = corpora.Dictionary(normalized_docs)
    saveDictionary(dictionary, 'brownDict.dict')

    brownCorpus = BrownCorpus(dictionary)
    saveCorpus(brownCorpus, 'brownCorpus_bow_norm.mm')

    tfidfModel = models.TfidfModel(brownCorpus)
    saveModel(tfidfModel, 'brown_tfidf_model.tfidf')

    tfidfCorpus = tfidfModel[brownCorpus]
    saveCorpus(tfidfCorpus, 'brown_tfidf_corpus.mm')

    ldaModel = models.LdaModel(tfidfCorpus, id2word=dictionary, num_topics=100, passes=20, iterations=400)
    saveModel(ldaModel, 'brown_lda_model.lda')

    ldaCorpus = ldaModel[tfidfCorpus]
    saveCorpus(ldaCorpus, 'brown_lda_corpus.mm')


def loadPretrainedLDACorpusAndModel():
    corpus = loadCorpus('brown_lda.corpus.mm')
    model = loadModel('brown_lda_model.lda', 'LDA')

    return corpus, model

    # corpus, model = topic_modeling.loadPretrainedLDACorpusAndModel()

    # print(model.print_topics(10))

    # for doc in corpus:
    #     print(doc)
