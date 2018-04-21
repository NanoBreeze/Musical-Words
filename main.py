from docstats import DocStats
from nltk.corpus import brown
from sentiment import WatsonAnalyzer
import normalization
import topic_modeling
import util

import json
from pprint import pprint
import logging


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    #words = normalizeRaw(raw)

    # normalized_docs = [normalization.normalizeWords(brown.words(fileid)) for  fileid in brown.fileids()]
    # topic_modeling.saveDictionary(normalized_docs, 'brownDict')

    # dictionary = topic_modeling.loadDictionary('brownDict.dict')
    # brownCorpus = topic_modeling.loadCorpus('brownCorpus_bow_norm.mm')
    # brownCorpus = topic_modeling.BrownCorpus(dictionary)

    # topic_modeling.saveCorpus(brownCorpus, 'brownCorpus_bow_norm')
    # brownCorpus = topic_modeling.loadDictionary('brownCorpus_bow_norm.mm')

    # topic_modeling.saveLDAModelForCorpus(brownCorpus, dictionary, 'brownLDA', 10)

    # ldaModel = topic_modeling.loadLDA('brownLDA.lda')

    # topic_modeling.initModelPipeline()
    corpus, model = topic_modeling.loadPretrainedLDACorpusAndModel()

    print(model.print_topics(10))

    # for doc in corpus:
    #     print(doc)






    '''
    raw = gutenberg.raw('austen-sense.txt')[:1000]
    username, password = util.getCredentials('credentials.json')
    text = 'Team, I know that times are tough! Product sales have been disappointing for the past three quarters. We have a competitive product, but we need to do a better job of selling it!'

    sentimentAnalyzer = WatsonAnalyzer(username, password)
    tone = sentimentAnalyzer.analyze(raw)

    print(json.dumps(tone, indent=2))
    '''

    '''
    raw = gutenberg.raw('austen-sense.txt')[:100]
    pprint(raw)
    '''



    '''
    raw = gutenberg.raw('austen-sense.txt')[:1000]

    stat = DocStats(raw)
    # stat.plot.wordLengthHistogram()
    stat.plot.sentenceLengthHistogram()

    input()
    '''
