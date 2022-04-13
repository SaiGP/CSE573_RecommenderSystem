"""
This custom sentiment analysis tool is based on an example given by Ashley Gingeleski.
It can be found at https://ashleygingeleski.com/2021/03/31/sentiment-analysis-of-product-reviews-with-python-using-nltk/
"""

import re
import numpy as np
import pandas as pd
import nltk
import json
import string
import pickle
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import scipy.sparse
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier

def tokenize_review(review):
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    tokenizer = WhitespaceTokenizer()

    return [current_token.strip(string.punctuation) for current_token in tokenizer.tokenize(review) if current_token not in stop_words]

def convert_to_word_freq_files():
    reviews_with_rating = ["", "", "", "", "", ""]
    with open("train_sentiment.json", "r") as reviews:
        count = 0
        for current_line in reviews:
            current_review = json.loads(current_line)
            reviews_with_rating[current_review["rating"]] += " " + current_review["review_text"].lower()
            count += 1
            print(count)


    stop_words = set(stopwords.words('english') + list(string.punctuation))
    tokenizer = WhitespaceTokenizer()

    rating_tokens = []
    for current_rating in range(1,6):
        print(current_rating, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        rating_tokens.append([current_token.strip(string.punctuation) for current_token in tokenizer.tokenize(reviews_with_rating[current_rating]) if current_token not in stop_words])
        frequency_temp = nltk.FreqDist(rating_tokens[current_rating - 1])
        with open("train_word_freq_rating_" + str(current_rating) + ".csv", "w+") as rating_file:
            for x in frequency_temp:
                rating_file.write(x + "," + str(frequency_temp[x]) + '\n')


def vectorizer(rating, start_count, end_count):
    reviews_with_rating = []
    with open("train_sentiment.json", "r") as reviews:
        count = 0
        for current_line in reviews:
            current_review = json.loads(current_line)
            if current_review["rating"] == rating:
                count += 1
                if start_count <= count:
                    reviews_with_rating.append(current_review["review_text"])
                    print(count)
                if count >= end_count:
                    break
    print(rating, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,3),tokenizer = token.tokenize, binary=True, max_features=1000)
    text_counts= cv.fit_transform(reviews_with_rating)
    pd.DataFrame(data=text_counts.toarray(), columns=cv.get_feature_names_out()).to_csv('train_word_count_rating_' + str(rating) + "_" + str(start_count) + "_" + str(end_count) + ".csv", index=False)

def frequency(rating, *file_paths):
    tokens = None
    occurrences = None
    current_count = 0

    for current_path in file_paths:
        with open(current_path, "r") as input_file:
            for current_line in input_file:
                if tokens is None:
                    tokens = current_line.split(",")
                elif occurrences is None:
                    occurrences = np.fromstring(current_line, dtype=float, sep=',')
                    current_count += 1
                else:
                    occurrences += np.fromstring(current_line, dtype=float, sep=',')
                    current_count += 1

    occurrences /= current_count

    with open(str(rating) + "freq.csv", "w+") as output_file:
        for current_token in range(len(tokens)):
            output_file.write(tokens[current_token] + "," + str(occurrences[current_token]) + "\n")

def get_token_of_length_with_freq(tokens, frequency, length):
    output = []
    for current_token, current_frequency in zip(tokens, frequency):
        if str(current_token).count(" ") + 1 == length:
            output.append((str(current_token), current_frequency))
    return output

def prune_same_words(tokens, frequency, minimum_independent_occurences):
    output = get_token_of_length_with_freq(tokens, frequency, 3)

    temp_list = []
    for current_two_length in get_token_of_length_with_freq(tokens, frequency, 2):
        current_frequency = current_two_length[1]
        for current_token, token_freq in output:
            temp_split = current_token.split(" ")
            current_split = current_two_length[0].split(" ")
            if temp_split[0] == current_split[0] and temp_split[1] == current_split[1]:
                current_frequency -= token_freq
                print(current_token, current_two_length[0])
            elif temp_split[1] == current_split[0] and temp_split[2] == current_split[1]:
                current_frequency -= token_freq
                print(current_token, current_two_length[0])

        if current_frequency >= minimum_independent_occurences:
            temp_list.append((current_two_length[0], current_frequency))
        else:
            print("Pruned", current_two_length)

    output.extend(temp_list)

    temp_list = []

    for current_one_length in get_token_of_length_with_freq(tokens, frequency, 1):
        current_frequency = current_one_length[1]
        for current_token, token_freq in output:
            if current_one_length[0] in current_token.split(" "):
                current_frequency -= token_freq
                print(current_token, current_one_length[0])

        if current_frequency >= minimum_independent_occurences:
            temp_list.append((current_one_length[0], current_frequency))
        else:
            print("Pruned", current_one_length)

    output.extend(temp_list)

    return output




def prune_frequencies():
    temp = pd.read_excel('rating_frequencies.xlsx', index_col=0, skiprows=lambda x: x in [0, 1],
                        sheet_name='Sheet1', usecols="A:O")
    rating_1_tokens = temp["Token"]
    rating_1_freq = temp["Frequency"]
    rating_2_tokens = temp["Token.1"]
    rating_2_freq = temp["Frequency.1"]
    rating_3_tokens = temp["Token.2"]
    rating_3_freq = temp["Frequency.2"]
    rating_4_1_tokens = temp["Token.3"]
    rating_4_1_freq = temp["Frequency.3"]
    rating_4_2_tokens = temp["Token.4"]
    rating_4_2_freq = temp["Frequency.4"]
    rating_5_1_tokens = temp["Token.5"]
    rating_5_1_freq = temp["Frequency.5"]
    rating_5_2_tokens = temp["Token.6"]
    rating_5_2_freq = temp["Frequency.6"]

    word_freq_dict = dict()

    for current_token, current_freq in prune_same_words(rating_1_tokens, rating_1_freq, 0.005):
        word_freq_dict[current_token] = (current_freq, 0.0, 0.0, 0.0, 0.0)

    for current_token, current_freq in prune_same_words(rating_2_tokens, rating_2_freq, 0.005):
        if current_token in word_freq_dict:
            temp = word_freq_dict[current_token]
            word_freq_dict[current_token] = (temp[0], current_freq, 0.0, 0.0, 0.0)
        else:
            word_freq_dict[current_token] = (0.0, current_freq, 0.0, 0.0, 0.0)

    for current_token, current_freq in prune_same_words(rating_3_tokens, rating_3_freq, 0.005):
        if current_token in word_freq_dict:
            temp = word_freq_dict[current_token]
            word_freq_dict[current_token] = (temp[0], temp[1], current_freq, 0.0, 0.0)
        else:
            word_freq_dict[current_token] = (0.0, 0.0, current_freq, 0.0, 0.0)

    for current_token, current_freq in prune_same_words(rating_4_1_tokens, rating_4_1_freq, 0.005):
        if current_token in word_freq_dict:
            temp = word_freq_dict[current_token]
            word_freq_dict[current_token] = (temp[0], temp[1], temp[2], current_freq, 0.0)
        else:
            word_freq_dict[current_token] = (0.0, 0.0, 0.0, current_freq, 0.0)

    for current_token, current_freq in prune_same_words(rating_4_2_tokens, rating_4_2_freq, 0.005):
        if current_token in word_freq_dict:
            temp = word_freq_dict[current_token]
            word_freq_dict[current_token] = (temp[0], temp[1], temp[2], (temp[3] + current_freq) / 2.0, 0.0)
        else:
            word_freq_dict[current_token] = (0.0, 0.0, 0.0, current_freq, 0.0)

    for current_token, current_freq in prune_same_words(rating_5_1_tokens, rating_5_1_freq, 0.005):
        if current_token in word_freq_dict:
            temp = word_freq_dict[current_token]
            word_freq_dict[current_token] = (temp[0], temp[1], temp[2], temp[3], current_freq)
        else:
            word_freq_dict[current_token] = (0.0, 0.0, 0.0, 0.0, current_freq)

    for current_token, current_freq in prune_same_words(rating_5_2_tokens, rating_5_2_freq, 0.005):
        if current_token in word_freq_dict:
            temp = word_freq_dict[current_token]
            word_freq_dict[current_token] = (temp[0], temp[1], temp[2], temp[3], (temp[4] + current_freq) / 2.0)
        else:
            word_freq_dict[current_token] = (0.0, 0.0, 0.0, 0.0, current_freq)

    with open("complexest_tokens_freq.csv", "w+") as output_file:
        for current_key in word_freq_dict:
            temp = word_freq_dict[current_key]
            output_file.write(current_key)
            for x in temp:
                output_file.write("," + str(x))
            output_file.write("\n")

class InputFormatter:
    def __init__(self):
        temp = pd.read_excel('complexest_tokens_freq.xlsx', index_col=None, dtype=str, header=0,
                             sheet_name='complexest_tokens_freq', usecols="A:A")
        temp["Token"].replace("'","")
        temp.sort_values(by=["Token"])
        self.tokens =  temp["Token"].values.tolist()
        self.tokens.sort()

        self.count_vectorizer = CountVectorizer(lowercase=True,ngram_range = (1,3),binary=True)
        self.count_vectorizer.fit(self.tokens)
        self.count_vectorizer.vocabulary_ = dict()
        for current in enumerate(self.tokens):
            self.count_vectorizer.vocabulary_[current[1]] = current[0]

    def build_dataset(self, input_path, output_path):
        with open(input_path, "r") as reviews:
            row_numbers = []
            col_numbers = []
            values = []

            count = 0

            for current_line in reviews:
                print(count)
                current_review = json.loads(current_line)
                current_rating = current_review["rating"]
                formatted_review = self.format_review(str(current_review["review_text"]))
                non_zero_indices = formatted_review.nonzero()

                row_numbers.extend([count] * (len(non_zero_indices[0]) + 1))
                col_numbers.extend(np.array(non_zero_indices).tolist()[0])
                col_numbers.append(len(self.tokens))
                values.extend([1] * len(non_zero_indices[0]))
                values.append(current_rating)
                count += 1
            temp = scipy.sparse.coo_matrix((values, (row_numbers, col_numbers)), shape=(count, len(self.tokens) + 1), dtype=int)
            scipy.sparse.save_npz(output_path, temp)


    def format_review(self, review):
        return self.count_vectorizer.transform([review]).toarray()[0]

    def format_reviews(self, reviews):
        return self.count_vectorizer.transform(reviews).toarray()

    def match_words(self, input):
        temp = []
        for x in range(len(input)):
            if input[x] == 1:
                temp.append(x)
        # print(temp)
        found_keys = []
        for current_key in self.count_vectorizer.vocabulary_:
            if self.count_vectorizer.vocabulary_[current_key] in temp:
                found_keys.append(current_key)
        print(found_keys)

def train_nb_classifier():
    a = scipy.sparse.load_npz("train_sentiment.npz").toarray()
    nb_classifier = CategoricalNB()
    nb_classifier.fit(a[:,:-1], a[:,-1])
    with open('nb_classifier.pkl', 'wb') as f:
        pickle.dump(nb_classifier, f)

def build_custom_nb_classifier(training_data):
    likelihoods = np.zeros((5, len(training_data[0]) - 1), dtype=np.longdouble)
    rating_likelihood = np.zeros(5, dtype=np.longdouble)

    for current_row in training_data:
        likelihoods[current_row[-1] - 1] += current_row[:-1]
        rating_likelihood[current_row[-1] - 1] += 1

    likelihoods = (likelihoods + 1) / (rating_likelihood + 2)[:,None]
    rating_likelihood /= training_data.shape[0]

    np.savetxt("nb_likelihoods.csv", np.log(likelihoods), delimiter=',')
    np.savetxt("nb_rating_likelihood.csv", np.log(rating_likelihood), delimiter=',')


class NaiveBayesClassifier:
    def __init__(self, likelihood_path, prior_path):
        self.likelihoods = np.loadtxt(likelihood_path, delimiter=",", dtype=np.longdouble)
        self.negative_likelihoods = np.log(1.0 - np.exp(self.likelihoods))
        self.prior = np.loadtxt(prior_path, delimiter=",", dtype=np.longdouble)

    def predict(self, input_values):
        rating_values = np.dot(self.likelihoods,input_values.T).T + self.prior
        rating_values += np.dot(self.negative_likelihoods, 1.0 - input_values.T).T
        return np.argmax(rating_values,axis=1) + 1

    def rating_likelihood(self, input_values):
        rating_values = np.dot(self.likelihoods,input_values.T).T + self.prior
        rating_values += np.dot(self.negative_likelihoods, 1.0 - input_values.T).T
        return rating_values / rating_values.sum()

    def rating_likelihoods(self, input_values):
        rating_values = np.dot(self.likelihoods,input_values.T).T + self.prior
        rating_values += np.dot(self.negative_likelihoods, 1.0 - input_values.T).T
        return rating_values / rating_values.sum(axis=1)[:,None]

def build_random_forest_classifier(train_data, pieces):
    csc_data = train_data.tocsc()
    sample_size = csc_data.shape[0] // pieces

    for current_piece in range(pieces - 1):
        print(current_piece)
        #10,457,477 (363716, 262059, 673933, 1974756, 3502284, 3680729)
        #  class_weight={1: 5, 2: 4, 3: 1, 4: 1, 5: 1}
        # class_weight="balanced_subsample"
        current_forest = RandomForestClassifier(n_estimators=20, class_weight={1: 10, 2: 5, 3: 0.01, 4: 0.006, 5: 0.006}
                                                , min_impurity_decrease=0.000000001, max_depth=300, max_samples=0.7)
        print()
        temp_sample = csc_data[current_piece * sample_size: (current_piece + 1) * sample_size].toarray()
        print(temp_sample[:, :-1].shape, temp_sample[:, -1].shape)
        current_forest.fit(temp_sample[:, :-1], temp_sample[:, -1])

        with open('rf_classifier' + str(current_piece) + '.pkl', 'wb') as rf_file:
            pickle.dump(current_forest, rf_file)

    final_temp_forest = RandomForestClassifier(n_estimators=20, class_weight={1: 10, 2: 5, 3: 0.01, 4: 0.006, 5: 0.006}
                                                , min_impurity_decrease=0.000000001, max_depth=300, max_samples=0.7)
    final_temp_sample = csc_data[sample_size * (pieces - 1):].toarray()
    print(final_temp_sample[:, :-1].shape, final_temp_sample[:, -1].shape)
    final_temp_forest.fit(final_temp_sample[:, :-1], final_temp_sample[:, -1])

    with open('rf_classifier' + str(pieces - 1) + '.pkl', 'wb') as rf_file:
        pickle.dump(final_temp_forest, rf_file)

    final_forest = None
    for current_piece in range(pieces):
        with open('rf_classifier' + str(current_piece) + '.pkl', 'rb') as rf_file:
            random_forest_classifier = pickle.load(rf_file)
            if final_forest is None:
                final_forest = random_forest_classifier
            else:
                final_forest.estimators_ += random_forest_classifier.estimators_
    final_forest.n_estimators = len(final_forest.estimators_)

    with open('rf_classifier.pkl', 'wb') as rf_file:
        pickle.dump(final_forest, rf_file)

def test_random_forest_classifier(test_data):
    with open('rf_classifier.pkl', 'rb') as rf_file:
        random_forest_classifier = pickle.load(rf_file)
        return random_forest_classifier.predict(test_data[:,:-1].toarray())


class RandomForestClassifier:
    def __init__(self):
        with open('rf_classifier.pkl', 'rb') as rf_file:
            self.trained_model = pickle.load(rf_file)

    def predict(self, target):
        return self.trained_model.predict(target)

# test = InputFormatter()

# token_encoding = test.format_review("This is such an awesome book. My friend suggested it and their suggestion turned out to be true.")
# print(token_encoding)
# test.match_words(token_encoding)
# test.build_dataset("train_sentiment.json", "train_sentiment.npz")
# a = scipy.sparse.load_npz("test_sentiment.npz").toarray()
# print(a[:,-1])
#
# clf = CategoricalNB()
# clf.fit(a[:,:-1], a[:,-1])
# print(clf.category_count_)

# train_nb_classifier()


# a = scipy.sparse.load_npz("train_sentiment.npz").tocsc()
# build_random_forest_classifier(a,50)
# a = scipy.sparse.load_npz("test_sentiment.npz").toarray()
# predictions = test_random_forest_classifier(a)
# build_custom_nb_classifier(a)
# temp = NaiveBayesClassifier("nb_likelihoods.csv", "nb_rating_likelihood.csv")
# print(temp.predict(a[:,:-1]))
# exit()
#
# predictions = temp.predict(a[:,:-1])
# absolute_error = np.abs(predictions - a[:,-1]).sum() / a.shape[0]
#
# conf_mat = np.zeros((6, 6))
# for actual, prediction in zip(a[:,-1], predictions):
#     conf_mat[actual, prediction] += 1
#
# print(absolute_error)
# np.savetxt("rf_conf_mat.csv", conf_mat, delimiter=",")

def most_positive_negative():

    formatter = InputFormatter()
    book_best_sentiment_representations = dict()

    with open("train_data.json", "r") as reviews:
        count = 1
        nb_classifier = NaiveBayesClassifier("nb_likelihoods.csv", "nb_rating_likelihood.csv")
        for current_line in reviews:
            for current_review in json.loads(current_line):
                print(count)
                count += 1
                temp = formatter.format_review(current_review['review_text'])
                likelihoods = nb_classifier.rating_likelihood(temp)
                max_likelihood = np.argmax(likelihoods)
                current_book = current_review['book_id']

                if current_book not in book_best_sentiment_representations:
                    temp_sent = [(0.0,None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
                    temp_sent[max_likelihood] = (likelihoods[max_likelihood], current_review['review_text'])
                    book_best_sentiment_representations[current_book] = temp_sent
                else:
                    current_book_best_sentiments = book_best_sentiment_representations[current_book]
                    if current_book_best_sentiments[max_likelihood][0] < likelihoods[max_likelihood]:
                        current_book_best_sentiments[max_likelihood] = (likelihoods[max_likelihood], current_review['review_text'])
                        book_best_sentiment_representations[current_book] = current_book_best_sentiments

    with open("sentiment_book_ratings.json", "w+") as output:
        json.dump(book_best_sentiment_representations, output)



# most_positive_negative()

# with open("sentiment_book_ratings.json", "r") as input:
#     temp = json.load(input)
#     print(temp["7462184"])

# with open("train_data.json", "r") as reviews:
#     count = 1
#     nb_classifier = NaiveBayesClassifier("nb_likelihoods.csv", "nb_rating_likelihood.csv")
#     for current_line in reviews:
#         for current_review in json.loads(current_line):
#             print(json.dumps(current_review, indent=4, sort_keys=True))
#             print()
#             print()
#             count += 1
#             if count > 10:
#                 exit()