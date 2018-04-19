import nltk
import string
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np



class _Ratio:

    def __init__(self, words, sents, taggedWords):

        totalCharCount = sum(len(word) for word in words)

        self._avgCharsPerWord = totalCharCount / len(words)
        self._avgWordsPerSentence = len(words) / len(sents)


        # tagging: NOUN, VERB, ADJ, DET, ADV, etc.
        tagFreqDist = nltk.FreqDist(tag for (word, tag) in taggedWords)

        self._verbNounRatio = tagFreqDist['VERB'] / tagFreqDist['NOUN']
        self._adverbVerbRatio = tagFreqDist['ADV'] / tagFreqDist['VERB']
        self._adjectiveNounRatio = tagFreqDist['ADJ'] / tagFreqDist['NOUN']


        # Modal words represent possibility.
        # Source: https://en.wikipedia.org/wiki/English_modal_verbs
        modalWords = ['can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must']
        modalWordCount = sum(word for word in words if word.lower() in modalWords)

        self._modelWordRatio = modalWordCount / len(words)


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

    def __init__(self, words, sents):
        self.words = words
        self.sents = sents

    def wordLengthHistogram(self):

        wordLengthCounter = Counter(len(word) for word in self.words)

        labels, values = zip(*(wordLengthCounter.items()))

        plt.bar(x=labels, height=values, tick_label=np.arange(len(labels)))
        plt.xlabel('# of Chars per Word')
        plt.show()
        #wordLengthFreqDist = nltk.FreqDist(len(word) for word in self.words)
        #wordLengthFreqDist.plot(title='Histogram of Number of Characters Per Word')

    def sentenceLengthHistogram(self):

        sentLengthCounter = Counter(len(nltk.word_tokenize(sent)) for sent in self.sents)

        labels, values = zip(*sentLengthCounter.items())
        plt.bar(x=labels, height=values)
        plt.xlabel('# of Words per Sentence')
        plt.show()

            # sentLengthFreqDist = nltk.FreqDist(len(nltk.word_tokenize(sent)) for sent in self.sents)
        #sentLengthFreqDist.plot(title='Histogram of Number of Words Per Sentence')


class DocStats:

    def __init__(self, doc):
        # Doc is not normalized
        # Remove punctuations, which are added into word_tokenize. NOTE: 's is it's own word so remove that too
        self.words = nltk.word_tokenize(doc)
        self.words = [word for word in self.words
                      if word not in string.punctuation
                      and word != "'s"]

        self.sents = nltk.sent_tokenize(doc)
        self.taggedWords = nltk.pos_tag(self.words, tagset='universal')

        self.ratio = _Ratio(self.words, self.sents, self.taggedWords)
        self.plot = _Plot(self.words, self.sents)

    def frequentBigramCollocations(self):
        pass


