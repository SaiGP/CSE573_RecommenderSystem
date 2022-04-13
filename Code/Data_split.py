import json
import random
from sklearn.model_selection import train_test_split as tts


def get_data_by_attrib(datapath="../Dataset/"):
    book_path = "book_ids.json"
    en_book_file = open(book_path, 'r+')
    book_ids = json.load(en_book_file)

    files = book_ids.keys()
    books = open("./book_data.json", "w+")
    rev_books = open("./review_data.json", "w+")

    for file in files:
        m = datapath + file
        ids = book_ids[file]
        genre = file[:-5].split("_",2)[2]
        print(genre)
        with open(m) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                if data["book_id"] in ids:
                    out_data = {"book_id": data["book_id"],
                                "title": data["title"],
                                "average_rating": data["average_rating"],
                                "is_ebook":data["is_ebook"],
                                "similar_books": data["similar_books"],
                                "format": data["format"],
                                "authors": data["authors"],
                                "genre": genre}
                    print(out_data)
                    out_json = json.dumps(out_data)
                    books.write(out_json)
                    books.write("\n")
                line = f.readline()
            f.close()

        r_file = file.replace("books", "reviews", 1)
        with open(datapath + r_file) as r_f:
            line = r_f.readline()
            while line:
                data = json.loads(line)
                if data["book_id"] in ids:
                    out_data = {"user_id": data["user_id"],
                                "book_id": data["book_id"],
                                "rating": data["rating"],
                                "review_text": data["review_text"],
                                "date_added": data["date_added"]}

                    print(out_data)
                    out_json = json.dumps(out_data)
                    rev_books.write(out_json)
                    rev_books.write("\n")
                line = r_f.readline()
            r_f.close()

    books.close()
    rev_books.close()
    return


def testtrain_samples(n=10):
    filename = "./review_data.json"
    userids = []
    data = []

    print("Loading review data...")
    fp = open(filename, "r")
    line = fp.readline()
    i = 1
    while line:
        print("reading line: " + str(i))
        rec = json.loads(line)
        userids.append(rec["user_id"])
        data.append(rec)
        line = fp.readline()
        i += 1


    print("selecting test ids..." + str(n) + "% of total data " + str(len(userids)))
    # test_ids = random.sample(userids, int(len(userids) * n / 100))
    data_train, data_test = tts(data, test_size=0.1, shuffle=True)

    test_file = open("./test_data.json", "w+")
    train_file = open("./train_data.json", "w+")

    print("Creating test and training file...")
    json.dump(data_train, train_file)
    json.dump(data_test, test_file)

    '''
    for j, d in enumerate(data):
        print("record number: " + str(j))
        out_data = json.dumps(d)
        if d["user_id"] in test_ids:
            test_file.write(out_data)
            test_file.write("\n")

        else:
            train_file.write(out_data)
            train_file.write("\n")
    '''
    fp.close()
    test_file.close()
    train_file.close()
    print("DONE")
    return


# get_data_by_attrib()
testtrain_samples()

