from flair.models import TextClassifier
from flair.data import Sentence

class flairModel:
    def __init__(self):
        self.classifier = TextClassifier.load('en-sentiment')

    def flair_prediction(self, target):
        classifier = self.classifier
        sentence = Sentence(target)
        classifier.predict(sentence)
        score = sentence.score
        if (sentence.tag == 'NEGATIVE'):
            score*= -1
        
        return ((score + 1)*5)/2
