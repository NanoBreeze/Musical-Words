from gensim.summarization import summarize, keywords

def extractiveSummarize(text, ratio=0.01):
    # Make sure that there are at least summarizer.INPUT_MIN_LENGTH (= 10) sentences in the text
    return summarize(text, ratio)

def getKeywords(text, ratio=0.01):
    return keywords(text, ratio, scores=True)