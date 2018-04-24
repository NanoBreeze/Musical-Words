# 𝅘𝅥𝅯𝅘𝅥𝅯 Musical Words 𝅘𝅥𝅯𝅘𝅥𝅯

Musical Words  uses text mining techniques to pair piano music (Chopin's 24 preludes) to English text documents. 
I built it to gain more experience working with text mining and machine learning.

## Features

* **Text Normalization and Transformation:** Including stopwords, punctuations, etc. Text is transformed to TF-IDF vector space

* **POS tagging & bigram collocations:** Statistics about the raw text is computed with NLTK

* **Text classification:** Two classification models are used: Naive Bayes and SVM (with stochastic gradient descent).
The training and testing data are from the 20NewsGroup dataset. GridSearchCV is used to find optimal hyperparameters. 
Naive Bayes has a F1 score of ~0.7. Implemented using scikit-learn.

* **Topic modeling:** Uses Latent Dirichlet Allocation on Reuters dataset. Implementation from Gensim.

* **Text summarization and keywords deduction:** The summarization is extractive and is implemented via Gensim

* **Sentiment analysis:** IBM Watson's Tone Analyzer is used because it offers more detailed analysis than the polarity analysis that many other libraries offers.
We examine five sentiments related to a text: "joy", "sadness", "anger" "fear", and "confidence". A Watson API credential is needed to run the sentiment analysis.

## Description
While the above features are implemented, the program relies on sentiment analysis and text statistics 
to choose which piano song to play for a given piece of text. Every piano song has been manually assigned a value for each of the five sentiments: "joy", "sadness", "anger",
"fear", "confidence" and the program plays the song with the lowest weighted Euclidean distance with respect to the text. 

The pretrained classifiers and topic models are in the pretrained/ directory. The piano songs should be put into the songs/ directory but actual MP3 recordings 
are not included in this package (I haven't received permission from the musicians). 

## Installation
To install dependencies, run: `$ pip install -r requirements.txt`
Remember to also register for a Watson API credential so that the program can apply sentiment analysis to the text.

To execute the program, run: `$ python main.py <TextFile>`

## Future Improvements
- Increase # of songs (there are only 24 Chopin preludes right now). Can also add more variety of songs, eg, 'jazz', 'vocal', 'chorals', etc.
- Integrate with a reading service, eg, Kindle, so that music automatically plays as the user reads, instead of having to manually run the python program 
for each text
- Allow songs to loop so that if it takes the user 10mins to read a document and the recommended song is only 3mins long, then the user can continue listening 
for as long as it takes to read the whole document instead of having the program stop playing music even though user isn't finished reading
- OR split the documents into ~750 word chunks (amount of time to read 3mins of text, which is the average length of a song) and play songs based on those chunks. 
This solves both the problem of the above point, and Watson's pre-condition that it only analyzes the first 1000 sentences (so we can't pass in a huge text at 
once anyways)
- Create a model that incorporates the other computed text stats (eg, # of collocations, verb:noun ratio, etc.) into finding the song to play. 
Right now, the model uses only sentiment analysis and avg word length.
