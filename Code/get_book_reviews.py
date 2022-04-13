import json

book_id = input("Enter Book ID: ")

with open("sentiment_book_ratings.json") as sentiment_book_reviews:
    reviews_per_book = json.load(sentiment_book_reviews)

    for count, review in enumerate(reviews_per_book[book_id]):
        print("\nSentiment Rating " + str(count + 1) + " example: \n", review[1])
    #
    # for current in reviews_per_book:
    #     temp = reviews_per_book[current]
    #     empty_review = False
    #     for x in range(5):
    #         if temp[x][1] is None:
    #             empty_review = True
    #             break
    #     if empty_review:
    #         continue
    #     print(current)