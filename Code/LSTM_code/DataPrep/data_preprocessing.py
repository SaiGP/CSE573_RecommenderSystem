import json
import numpy as np
from datetime import datetime
from time import sleep


def data_generations(infile, outfile):
    """
    Reads infile data which is a list of dictionary items json file.
    for every user_id a sequence data consisting of User id, next book, rating and time stamp is created.
    The history for each of the current item is also stored.
    negative labels (0) are created for each of the records. This doubles the number of training, validation and testing samples.

    :param infile:
    :param outfile:
    :return:

    """

    infp = open(infile, "r")
    indata = json.load(infp)
    size = len(indata)
    data = {}
    item_ids = np.loadtxt("../iids.out", dtype="str")
    uiids = np.unique(item_ids)

    for ind, d in enumerate(indata):
        print("%d / %d" % (ind, size))
        user_id = d["user_id"]
        item_id = d["book_id"]
        rate = d["rating"]
        timestamp = int(datetime.strptime(d["date_added"], '%a %b %d %H:%M:%S %z %Y').timestamp())
        if user_id in data.keys():
            _create_sample(d, data[user_id], outfile, uiids)
        else:
            data[user_id] = [[], [], []]
        # print(item_id)
        data[user_id][0].append(item_id)
        data[user_id][1].append(rate)
        data[user_id][2].append(timestamp)

        '''print(data[user_id][0])
        print(data[user_id][1])
        print(data[user_id][2])'''

    print("DONE")
    infp.close()
    return


def _create_sample(d, data, file, uiids):
    """
    Takes a dictionary d, list of history items for a given user and creates sequence data with
    label 1 for positive sample and 0 for negative sample.
    the data is written into file.

    negative sample is created by randomly choosing a book that is not the next one in the sequence.

    :param d:
    :param data:
    :param file:
    :param uiids:
    :return:
    """
    outfp = open(file, "a+")
    uiids = uiids[uiids != d["book_id"]]
    timestamp = int(datetime.strptime(d["date_added"], '%a %b %d %H:%M:%S %z %Y').timestamp())

    # positive sample.
    item_hst = ",".join(str(s) for s in data[0])
    rate_hst = ",".join(str(s) for s in data[1])
    timestp_hst = ",".join(str(s) for s in data[2])
    outfp.write("1\t" +
                d["user_id"] + "\t" +
                d["book_id"] + "\t" +
                str(d["rating"]) + "\t" +
                str(timestamp) + "\t" +
                item_hst + "\t" +
                rate_hst + "\t" +
                timestp_hst + "\n")

    # negative sample
    item = np.random.choice(uiids, size=1)
    # print(item)
    outfp.write("0\t" +
                d["user_id"] + "\t" +
                str(item[0]) + "\t" +
                str(d["rating"]) + "\t" +
                str(timestamp) + "\t" +
                item_hst + "\t" +
                rate_hst + "\t" +
                timestp_hst + "\n")

    # print("added +ve and -ve dataset lines")

    outfp.close()
    return


# data_generations("test_data.json", "test.out")

#print("validation data")
#data_generations("valid_data.json", "valid.out")

#print("training data...")
#data_generations("train8_data.json", "train.out")

def sequencing(data_dict, file):
    out_fp = open(file, "a+")

    for k in data_dict.keys():
        books = data_dict[k][0]
        rating = data_dict[k][1]
        timestamp = data_dict[k][2]
        print(len(rating))

        while len(rating) > 10:
            book = str(books.pop())
            rate = str(rating.pop())
            ts = str(timestamp.pop())
            book_hst = ",".join(str(s) for s in books[-10:])
            rate_hst = ",".join(str(s) for s in rating[-10:])
            ts_hst = ",".join(str(s) for s in timestamp[-10:])
            print(len(rate_hst), len(rating))
            out_fp.write(k + "\t" +
                         book + "\t" +
                         rate + "\t" +
                         ts + "\t" +
                         book_hst + "\t" +
                         rate_hst + "\t" +
                         ts_hst + "\n")
    out_fp.close()
    return

def create_seq(infile, outfile):
    infp = open(infile, "r")
    indata = json.load(infp)
    dfield = {}  # "user_id": [[], [], []]

    print("Creating data dict...")
    for data in indata:
        user = data["user_id"]
        if user not in dfield.keys():
            dfield[user] = [[], [], []]

        dfield[user][0].append(data["book_id"])
        dfield[user][1].append(data["rating"])
        unixtime = int(datetime.strptime(data["date_added"], '%a %b %d %H:%M:%S %z %Y').timestamp())
        dfield[user][2].append(unixtime)

    print("Data Processing")
    kval = list(dfield.keys())[0]
    print(type(dfield[kval][0]), type(dfield[kval][1]), type(dfield[kval][2]))

    delete_key = [k for k in dfield if len(dfield[k][0]) <= 10]
    for key in delete_key:
        del dfield[key]

    for k in dfield.keys():
        ind = np.array(dfield[k][2]).argsort()
        for i in range(3):
            fld_np = np.array(dfield[k][i])
            fld_np = fld_np[ind]
            dfield[k][i] =fld_np.tolist()
    kval = list(dfield.keys())[0]
    print(type(dfield[kval][0]), type(dfield[kval][1]), type(dfield[kval][2]), len(dfield.keys()))

    sequencing(dfield, outfile)

    print("DONE!!!")
    return


#create_seq("test_data.json", "test_seq.out")
# create_seq("train8_data.json", "train_seq.out")
#create_seq("valid_data.json", "valid_seq.out")

fp = open("../../books_list.json", "r")
data = json.load(fp)
iids = []

for d in data:
    iids.append(d["book_id"])

uiids = np.array(iids)
uiids = np.unique(uiids)

np.savetxt("uiids.out", uiids, fmt="%s", delimiter=',')




