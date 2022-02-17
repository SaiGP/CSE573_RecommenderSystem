import os
import json
import random

datapath = "../Dataset/"
rep_regex = r'}[\ \r\n"\r\n"]*{'


def readjson():
    files = os.listdir(datapath)
    records = [["title", "records", "Keys_list"]]

    for file in files:
        m = datapath + file
        print(m)
        cnt = 0
        with open(m) as f:
            line = f.readline()
            data = json.loads(line)
            cnt = 1
            while line:
                line = f.readline()
                cnt += 1
            records.append([file, cnt, data.keys()])
            print(file + str(cnt))

    with open("records.csv", "w") as rec:
        for r in records:
            rec.write("%s\n" % r)

    return


def enbooks():
    files = os.listdir(datapath)
    # create a dict of list of book ids that have only english & save to file.
    book_list = {}
    ids = []
    for file in files:
        m = datapath + file
        print(m)
        cnt = 0
        if "books" in file:
            print("In books")
            ids.clear()

            with open(m) as f:
                line = f.readline()
                while line:
                    data = json.loads(line)
                    if data["language_code"] == "eng" or data["language_code"] == "":
                        ids.append(data["book_id"])
                        cnt += 1
                    line = f.readline()

            book_list[file] = ids.copy()
            print("TITLE: " + file + "with total english records: " + str(cnt))

    # write to json
    with open("book_ids.json", 'w+') as bi:
        bi.write(json.dumps(book_list))


def rand_sample(n=10):
    book_path = "book_ids.json"
    en_book_file = open(book_path, 'r+')
    book_ids = json.load(en_book_file)
    sample_ids = {}
    ids = []
    r_ids = []
    name = os.listdir(datapath)
    files = book_ids.keys()
    for file in files:
        ids.clear()
        r_ids.clear()

        print("Reading file: " + file)
        count = 0
        ids = book_ids[file]
        r_ids = random.sample(ids, int(len(ids) * n / 100))
        print(len(ids), len(r_ids))
        sample_ids[file] = r_ids.copy()

        with open(datapath + file, 'r') as book, open("./" + file, "w+") as out_book:
            book_text = book.readline()
            while book_text:
                text = json.loads(book_text)
                if text['book_id'] in r_ids:
                    out_book.write(book_text)
                    count += 1
                book_text = book.readline()
        print("Sample file %s created with %d records" % (file, count))
        count = 0
        book.close()
        out_book.close()

        file2 = file.replace("books", "reviews", 1)
        print("file2 as review file: ", file2)
        with open(datapath + file2, 'r') as review, open("./" + file2, "w+") as out_rev:
            rev_text = review.readline()
            while rev_text:
                text = json.loads(rev_text)
                if text['book_id'] in r_ids:
                    out_rev.write(rev_text)
                    count += 1
                rev_text = review.readline()

        print("Sample file %s created with %d records" % (file2, count))
        review.close()
        out_rev.close()

    with open("./sample_ids.json", "w+") as smids:
        smids.write(json.dumps(sample_ids))
    print("Sample files creation DONE! %d of the original files" % n)

def check_book_ids():
    file = open("./book_ids.json", 'r')
    data = json.load(file)
    # print(data.keys[])
    for d in data.keys():
        print(d+": " + str(len(data[d])))

# readjson()
# enbooks()
# check_book_ids()
rand_sample(10)
