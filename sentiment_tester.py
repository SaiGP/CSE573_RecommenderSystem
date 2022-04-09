import json
import numpy as np
from textblob_sentiment_predicter import text_blob_prediction
from vader_sentiment_predicter import VaderSentiment

def sentiment_tester(*sentiment_functions):
    absolute_errors = np.zeros(len(sentiment_functions))
    review_count = 0.0
    confusion_matrix = []
    for _ in range(len(sentiment_functions)):
        confusion_matrix.append(np.zeros((6,6)))

    with open("test_sentiment.json", "r") as reviews:
        for current_line in reviews:
            print(review_count)
            current_review = json.loads(current_line)
            for current_count, current_func in enumerate(sentiment_functions):
                prediction = current_func(current_review["review_text"])
                absolute_errors[current_count] += abs(prediction - current_review["rating"])
                confusion_matrix[current_count][current_review["rating"],round(prediction)] += 1
            review_count += 1

        absolute_errors /= review_count
        for current_count, current_func in enumerate(sentiment_functions):
            print(str(current_func),":")
            print("Average Absolute Error: ", absolute_errors[current_count])
            print("Confusion matrix (actual x predicted):")
            print(confusion_matrix[current_count])
            print("===================================")


# temp = VaderSentiment()
sentiment_tester(temp.predict_sentiment,text_blob_prediction)

