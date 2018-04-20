from docstats import DocStats
from nltk.corpus import gutenberg
from sentiment import WatsonAnalyzer
from normalization import normalizeRaw
import util

import json
from pprint import pprint



if __name__ == '__main__':

    '''
    raw = gutenberg.raw('austen-sense.txt')[:1000]
    username, password = util.getCredentials('credentials.json')
    text = 'Team, I know that times are tough! Product sales have been disappointing for the past three quarters. We have a competitive product, but we need to do a better job of selling it!'

    sentimentAnalyzer = WatsonAnalyzer(username, password)
    tone = sentimentAnalyzer.analyze(raw)

    print(json.dumps(tone, indent=2))
    '''

    raw = gutenberg.raw('austen-sense.txt')[:100]
    pprint(raw)


    pprint(normalizeRaw(raw))


    '''
    raw = gutenberg.raw('austen-sense.txt')[:1000]

    stat = DocStats(raw)
    # stat.plot.wordLengthHistogram()
    stat.plot.sentenceLengthHistogram()

    input()
    '''
