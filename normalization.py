import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import util

# Remove stopwords, special symbols & punctuations, capitalization, expand contractions, stemming, and spelling



def expandContractions(raw, contractions_map):
    expanded_raw = raw
    for contracted, expanded in contractions_map:
        expanded_raw = expanded_raw.replace(contracted, expanded)

    return expanded_raw


def removePunctuationsAndSpecialSymbols(words):
    filtered_words = [word for word in words if word[0] not in string.punctuation
                                                and word.isalpha()]
    # Check first character instead of "word not in string.punctuation" because sometimes, the symbols appears first.
    # Eg, Cathy's toy => ["Cathy", "'s"] and we want to remove the whole word "'s"
    # Checking if the word is alphanumeric is superset of punctuation. It ensures numbers aren't included
    # Why not jsut remove punctuation then? For now, keep both.

    return filtered_words


def removeStopWords(words):
    stopwds = stopwords.words('english') + util.CUSTOM_STOP_WORDS

    filtered_words = [word for word in words if word not in stopwds]

    return filtered_words


def stemWords(words):
    stemmer = SnowballStemmer('english')
    filtered_words = [stemmer.stem(word) for word in words]

    return filtered_words

def normalizeRaw(raw):
    """@:param: raw is a string. Eg, 'Once upon a time, there was a bunny'
    Returns a list of words"""
    raw = raw.lower()
    raw = expandContractions(raw, util.CONTRACTION_MAP)
    words = nltk.word_tokenize(raw)
    words = removePunctuationsAndSpecialSymbols(words)
    words = removeStopWords(words)
    words = stemWords(words)

    return words

def normalizeWords(words):
    """@:param: words is a list of words. Eg, ['Once', 'upon', 'a', 'time', ...]
    Sometimes calling a corpus' raw() function returns the words and their POS values
    This function is basically a wrapper around normalizeRaw"""

    # I know, this is taking the lazy way, and a bit less efficient
    raw = ' '.join(words)
    normalizedWords = normalizeRaw(raw)

    return normalizedWords

