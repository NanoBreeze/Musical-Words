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
    filtered_words = [word for word in words if word[0] not in string.punctuation]
    # Check first character instead of "word not in string.punctuation" because sometimes, the symbols appears first.
    # Eg, Cathy's toy => ["Cathy", "'s"] and we want to remove the whole word "'s"

    return filtered_words


def removeStopWords(words):
    stopwds = stopwords.words('english')
    filtered_words = [word for word in words if word not in stopwds]

    return filtered_words


def stemWords(words):
    stemmer = SnowballStemmer('english')
    filtered_words = [stemmer.stem(word) for word in words]

    return filtered_words

def normalizeRaw(raw):
    raw = raw.lower()
    raw = expandContractions(raw, util.CONTRACTION_MAP)
    words = nltk.word_tokenize(raw)
    words = removePunctuationsAndSpecialSymbols(words)
    words = removeStopWords(words)
    words = stemWords(words)

    return words

# Remove stopwords, special symbols & punctuations, capitalization, expand contractions, stemming, and spelling

#raw = "How're you doing today? I run quickly now. I earned 100.50$. I've been doing well lately. Don't do that. That's Cathy's toy. The car is big"


# nltk.word_tokenize(raw)

# string.punctuation
# stopwords.words('english') # all lowercase

# print(CONTRACTION_MAP)
