import os
import numpy as np
import playsound

# each entry is of the form: [joy, sadness, anger, fear, confident, isSmooth(Boolean)]
SONG_WEIGHTINGS = {
    'Prelude1' : (np.array([0.4, 0.1, 0.1, 0.0, 0.2]), False),
    'Prelude2' : (np.array([0.0, 0.8, 0.3, 0.5, 0.7]), True),
    'Prelude3' : (np.array([0.5, 0.2, 0.1, 0.0, 0.4]), True),
    'Prelude4' : (np.array([0.1, 0.9, 0.2, 0.3, 0.7]), True),
    'Prelude5' : (np.array([0.8, 0.1, 0.0, 0.2, 0.6]), True),
    'Prelude6' : (np.array([0.2, 0.7, 0.3, 0.3, 0.5]), False),
    'Prelude7' : (np.array([0.0, 0.8, 0.1, 0.1, 0.3]), False),
    'Prelude8' : (np.array([0.6, 0.2, 0.1, 0.7, 0.2]), True),
    'Prelude9' : (np.array([0.2, 0.7, 0.8, 0.3, 0.4]), False),
    'Prelude10': (np.array([0.9, 0.0, 0.0, 0.2, 0.9]), True),
    'Prelude11': (np.array([0.9, 0.0, 0.0, 0.0, 0.8]), True),
    'Prelude12': (np.array([0.6, 0.0, 0.2, 0.3, 1.0]), False),
    'Prelude13': (np.array([0.1, 0.7, 0.0, 0.2, 0.7]), True),
    'Prelude14': (np.array([0.2, 0.1, 0.8, 0.6, 0.8]), True),
    'Prelude15': (np.array([0.4, 0.6, 0.0, 0.0, 0.5]), True),
    'Prelude16': (np.array([0.3, 0.1, 0.8, 0.5, 0.9]), True),
    'Prelude17': (np.array([0.2, 0.4, 0.0, 0.1, 0.4]), True),
    'Prelude18': (np.array([0.2, 0.3, 0.4, 0.7, 0.8]), True),
    'Prelude19': (np.array([0.7, 0.0, 0.0, 0.1, 0.6]), False),
    'Prelude20': (np.array([0.0, 1.0, 0.8, 0.4, 1.0]), False),
    'Prelude21': (np.array([0.3, 0.5, 0.4, 0.3, 0.2]), False),
    'Prelude22': (np.array([0.0, 0.2, 0.7, 1.0, 0.8]), False),
    'Prelude23': (np.array([0.5, 0.1, 0.0, 0.0, 0.3]), True),
    'Prelude24': (np.array([0.1, 0.5, 0.8, 0.4, 0.8]), True)
}

def selectSong( avgWordsPerSentence, joy=0, sadness=0, anger=0, fear=0, confident=0):
    '''Each text contains five sentiments: joy, sadness, anger, fear and confident.
    The avgWordsPerSentence of a text represents whether the text is 'smooth' or not
    We want to select the song whose sentiments have the closest Euclidean distance to the text's sentiments
    However, if a text is smooth and a song is also smooth, decrease the Euclidean distance by 0.1.
    The same goes for not smooths
    '''

    songDir = './songs/'

    textWeighting = np.array([joy, sadness, anger, fear, confident])
    textIsSmooth = True if avgWordsPerSentence >20 else False

    closestSong = ""
    minDistance = float('inf')

    for currSong, weightings in SONG_WEIGHTINGS.items():
        musicWeighting = weightings[0]
        songIsSmooth = weightings[1]

        currNorm = np.linalg.norm(musicWeighting - textWeighting)

        if textIsSmooth == songIsSmooth:
            currNorm -= 0.1

        if currNorm < minDistance:
            minDistance, closestSong = currNorm, currSong

    return os.path.join(songDir, closestSong + '.mp3')


def play(path):
    playsound.playsound(path, True)