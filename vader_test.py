from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json


def vader_rating_prediction(sentence):
    sentiment_predictor = SentimentIntensityAnalyzer()
    review_sentiment = sentiment_predictor.polarity_scores(sentence)
    output = ((review_sentiment['neg'] * 1.0) + (review_sentiment['neu'] * 3.0) + (review_sentiment['pos'] * 5.0))
    return (output + 0.0000000000001) / (review_sentiment['neg'] + review_sentiment['neu']+ review_sentiment['pos'] + 0.0000000000001)

with open("test_sentiment.json", "r") as reviews:
    sentiment_predictor = SentimentIntensityAnalyzer()
    absolute_error = 0.0
    count = 0.0
    for current_line in reviews:
        current_review = json.loads(current_line)
        predicted_rating = vader_rating_prediction(current_review["review_text"])
        print(predicted_rating, current_review["rating"])
        absolute_error += abs(predicted_rating - current_review["rating"])
        count += 1.0
    print("Average Absolute Error:", absolute_error / count)
