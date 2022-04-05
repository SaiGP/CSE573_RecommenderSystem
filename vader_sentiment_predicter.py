from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VaderSentiment:
    def __init__(self):
        self.sentiment_predictor = SentimentIntensityAnalyzer()

    def predict_sentiment(self, target):
        review_sentiment = self.sentiment_predictor.polarity_scores(target)
        output = ((review_sentiment['neg'] * 1.0) + (review_sentiment['neu'] * 3.0) + (review_sentiment['pos'] * 5.0))
        return (output + 0.0000000000001) / (review_sentiment['neg'] + review_sentiment['neu'] + review_sentiment['pos'] + 0.0000000000001)
