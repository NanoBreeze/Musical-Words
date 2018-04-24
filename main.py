from docstats import DocStats
from nltk.corpus import brown
from nltk.corpus import gutenberg
from sentiment import WatsonAnalyzer
import normalization
import topic_modeling
import util
import classification
import songs

import json
from pprint import pprint
import logging
import pdb
import sys


if __name__ == '__main__':

    raw = 'Team, I know that times are tough! Product sales have been disappointing for the past three quarters. We have a competitive product, but we need to do a better job of selling it!'
    try:
        file = sys.argv[1]
        with open(file, 'r') as f:
            raw = f.read() # Watson allows only 1000 sentences and submissions of less than 128KB. Here, we assume the text fulfills both requirements
    except:
        pass

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    username, password = util.getCredentials('credentials.json')

    stat = DocStats(raw)
    sentimentAnalyzer = WatsonAnalyzer(username, password)
    tone = sentimentAnalyzer.analyze(raw)

    pprint(tone)

    songPath = songs.selectSong(stat.ratio.avgWordsPerSentence(), joy=tone['joy'], sadness=tone['sadness'], anger=tone['anger'], fear=tone['fear'], confident=tone['confident'])
    songs.play(songPath)
