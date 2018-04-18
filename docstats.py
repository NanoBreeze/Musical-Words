import nltk



class _Ratio:

    def __init__(self, doc):
        self.words = nltk.word_tokenize(doc)
        self.sents = nltk.sent_tokenize(doc)
        self.taggedWords = nltk.pos_tag(self.words)

        self.computeRatios(doc)

    def computeRatios(self, doc):



    def avgCharsPerWord(self):
        return self._avgCharsPerWord

    def avgWordsPerSentence(self):
        return self._avgWordsPerSentence

    def verbNounRatio(self):
        return self._verbNounRatio

    def adverbVerbRatio(self):
        return self._adverbVerbRatio

    def adjectiveNounRatio(self):
        return self._adjectiveNounRatio

    def modelWordRatio(self):
        return self._modelWordRatio


class _Plot:

    def wordLengthHistogram(self):
        pass

    def sentenceLengthHistogram(self):
        pass


class DocStats:

    def __init__(self, doc):
        self.doc = doc
        self.ratio = _Ratio(doc)

    def frequentBigramCollocations(self):
        pass


