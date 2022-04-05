from textblob import TextBlob

def text_blob_prediction(target):
    review_sentiment = TextBlob(target).sentiment.polarity
    return (2.0 * review_sentiment) + 3.0