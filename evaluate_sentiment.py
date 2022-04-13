from vader_sentiment_predicter import VaderSentiment
from textblob_sentiment_predicter import text_blob_prediction
from custom_sentiment_analysis import InputFormatter, NaiveBayesClassifier, RandomForestClassifier
import json
import sys
import numpy as np

formatter = InputFormatter()
vader_sentiment = VaderSentiment()
naive_bayes_classifier = NaiveBayesClassifier("nb_likelihoods.csv", "nb_rating_likelihood.csv")
random_forest_classifier = RandomForestClassifier()

np.set_printoptions(threshold=sys.maxsize)
review = input("Enter Review: ")

encoding = formatter.format_review(review)

print(encoding)

formatter.match_words(encoding)

print("VADER Sentiment: ", vader_sentiment.predict_sentiment(review))
print("TextBlob Sentiment: ", text_blob_prediction(review))
print("-----------------------------------------------------------------")
print("Naive Bayes Classifier Sentiment: ", naive_bayes_classifier.predict(encoding.reshape((1, len(encoding)))))
print("Random Forest Classifier Sentiment: ", random_forest_classifier.predict(encoding.reshape((1, len(encoding)))))

