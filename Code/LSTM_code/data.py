import gc
import json
import os
import random
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import numpy
import numpy as np

datapath = "../"


def createUserseq():

    print("Data processing user sequences...")
    train_file = datapath + "train_data.json"
    train_fp = open(train_file, "r")
    train_data = json.load(train_fp)
    user_list =[]
    item_list = []
    rating_list = []
    date_list = []
    for td in train_data:
        user_list.append(td["user_id"])
        item_list.append(td["book_id"])
        rating_list.append(td["rating"])
        date_list.append(int(datetime.strptime(td["date_added"], '%a %b %d %H:%M:%S %z %Y').timestamp()))

    # arg sort the list then combine by user for hot encoding. dataframe or dict.

    user_ids = np.array(user_list)
    item_ids = np.array(item_list)
    rating = np.array(rating_list)
    timestamp = np.array(date_list)

    ind_sorted = np.argsort(timestamp)
    timestamp = timestamp[ind_sorted]
    rating = rating[ind_sorted]
    item_ids = item_ids[ind_sorted]
    user_ids = user_ids[ind_sorted]

    print(user_ids[0])
    print(date_list[0])

    np.savetxt("rating.out", rating, delimiter=',', fmt='%f')
    np.savetxt("timestamp.out", timestamp, delimiter=',', fmt='%d')
    np.savetxt("iids.out", item_ids, delimiter=',', fmt='%s')
    np.savetxt("uids.out", user_ids, delimiter=',', fmt='%s')

    train_fp.close()
    gc.collect()


    return user_ids, item_ids, rating, timestamp


def dataEncoding():
    path = "./"
    files = os.listdir(path)

    if "iids.out" in files:
        item_ids = np.loadtxt(path+"iids.out", delimiter=',', dtype='str')

    print("User data ...")

    if not np.size(item_ids):
        print("creating users sequence")
        user_ids, item_ids, rating, timestamp = createUserseq()

    uiids = np.unique(item_ids)
    ids = []

    for u in uiids:
        ids.append([u, random.randint(0,5)])

    le_item = OneHotEncoder(handle_unknown='ignore')
    le_item.fit(ids)
    """le_rate = LabelEncoder()
    rate_encode = le_rate.fit(urate)"""

    print(ids[:5])

    # print(item_encode[:5])
    itms = [['17902034', 5], ['9609208', 5], ['17450549', 5]]

    val = le_item.transform(itms).toarray()
    z = [val[1]]

    """l = uiids[:10]
    print(len(l))
    print(le_item.transform(l))"""
    print(val)
    # creating sequence for users...
    print(le_item.inverse_transform(z))
    # encode sequence for users...

    return


# dataEncoding()

def createitemslist():

    print("Data processing user sequences...")
    train_file = datapath + "train_data.json"
    train_fp = open(train_file, "r")
    train_data = json.load(train_fp)

    item_list = []
    for td in train_data:
        item_list.append(td["book_id"])

    test_file = datapath + "test_data.json"
    test_fp = open(test_file, "r")
    test_data = json.load(test_fp)
    for td in test_data:
        item_list.append(td["book_id"])

    item_ids = np.array(item_list)

    np.savetxt("iids.out", item_ids, delimiter=',', fmt='%s')

    train_fp.close()
    gc.collect()


createitemslist()


def split_file(file):
    f = open(file, "r")
    cnt = 0
    line = f.readline()

    for i in range(10):
        outfile = file.split(".")[0]+str(i)+".out"
        outf = open(outfile, "w+")
        while line:
            outf.write(line)
            line = f.readline()
            cnt += 1
            if cnt >= 100000:
                break
    return

