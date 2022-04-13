# CSE579_RecommenderSystem

**Dataset**:

Good reads: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home

Main dataset link: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home

Gdrive link: https://drive.google.com/drive/folders/1Dwh7KL0EGTq-zjTzy6rzMdBlcTJKI-eL?usp=sharing

10% data link: https://drive.google.com/drive/folders/1knNVLlO62on75TGaCvgYGQhoTiLd7Y_M?usp=sharing

*NOTE:* use readline for the files as given in the defalut code. import json and use loads/dumps to read the data.

Data for out use:
1. books_data.json: fields {"book_id": data["book_id"],
                            "title": data["title"],
                            "average_rating": data["average_rating"],
                            "is_ebook":data["is_ebook"],
                            "similar_books": data["similar_books"],
                            "format": data["format"],
                            "authors": data["authors"],
                            "genre": genre}
   
   LINK: https://drive.google.com/file/d/1bT46RdLCY6aTDSKpYxNJKAScEGGp8vxP/view?usp=sharing
                            
2. review_data.json: fields {"user_id": data["user_id"],
                              "book_id": data["book_id"],
                              "rating": data["rating"],
                              "review_text": data["review_text"],
Contains all the users reviews for all the english books. Total 11619419 records.

3. train_data.json: (90% review data)


4. test_data.json: (10% review data) https://drive.google.com/file/d/17NUNc0vnBTBG3CQWdwd_deW5puaV7Kkz/view?usp=sharing


5. Sentiment_Analysis_Data.rar https://drive.google.com/file/d/1aOaTEsaCNNmTcmHBa1yCOOMTal1BTaJS/view?usp=sharing
* complexest_tokens_freq.xlsx: Word frequency Excel Document
* nb_likelihoods.csv: likelihood percentages for the trained Naive Bayes Classifier
* nb_rating_likelihoods.csv: rating priors for the trained Naive Bayes Classifier
* rf_classifier.pkl: pickeled trained random forest model
* sentiment_book_ratings.json: Top sentiment ratings for each rating for each book in the training dataset
* test_sentiment.json: testing dataset for the sentiment analysis models
* train_sentiment.json: training dataset for the sentiment analysis models 


6. Google Drive Link https://drive.google.com/drive/folders/1Dwh7KL0EGTq-zjTzy6rzMdBlcTJKI-eL?usp=sharing
