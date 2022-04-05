from happytransformer import HappyTextClassification

#Do not use! It has a 512 token limit which is too small for some of the reviews.

class HappyTransformerSentiment:
    def __init__(self):
        self.sentiment_predictor = HappyTextClassification(model_type="DISTILBERT",
                                                           model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                                           num_labels=2)
    def predict_sentiment(self, target):
        review_sentiment = self.sentiment_predictor.classify_text(target)
        if review_sentiment.label == "POSITIVE":
            return 3.0 + (2.0 * review_sentiment.score)
        elif review_sentiment.label == "NEGATIVE":
            return 3.0 - (2.0 * review_sentiment.score)
        else:
            return 3.0