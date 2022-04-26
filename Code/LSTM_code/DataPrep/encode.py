import sys
import csv
import json
import os.path
from time import sleep

import numpy as np
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from random import randint
import _pickle as cPickle



def dataEncoding(infile, outfile_x, outfile_y):
    print("User data ...")
    # get all the item ids

    if os.path.exists("./fit_seq.out"):
        print("Loading fit sequence ...")
        seq = open("./fit_seq.out", "r")
        stmts = csv.reader(seq)
        fseq = list(stmts)
        fit_seq = [[u[0], int(u[1])] for u in fseq]

    else:
        print("Creating fit sequence...")
        item_ids = np.loadtxt("../iids.out", delimiter=',', dtype='str')
        uiids = np.unique(item_ids)
        fit_seq = [[u, randint(0, 5)] for u in uiids] # generate sequences to predict item, rating value
        with open("./fit_seq.out", "w", newline="") as outf:
            fitwrite = csv.writer(outf)
            fitwrite.writerows(fit_seq)


    print(fit_seq[0])
    # setup the encoder
    ohe_item = OneHotEncoder()
    ohe_item.fit(fit_seq)
    print(ohe_item.categories_)
    # open test/training/validation files
    file = open(infile, "r")
    outfile = open(outfile_x, "wb")
    xpickle = cPickle.Pickler(outfile)
    yfile = open(outfile_y, "wb")
    ypickle = cPickle.Pickler(yfile)

    line = file.readline()
    i = 1
    cnt = 0

    while line:
        line = file.readline()
        i += 1
        print("read %dth line..." % i)
        data = line.split("\t")
        if int(data[0]) == 0:
            continue

        item_hist = data[5].split(",")
        if len(item_hist) >= 5:
            cnt += 1
            rate_hist = [int(i) for i in data[6].split(",")]
            x_data = [list(d) for d in zip(item_hist, rate_hist)]
            print(x_data)
            enc_val = ohe_item.transform(x_data)
            print(enc_val.toarray())
            xpickle.dump(enc_val)
            y_data = [[data[2], int(data[3])]]
            print(y_data)
            sleep(1)
            enc_yval = ohe_item.transform(y_data)
            ypickle.dump(enc_yval)

    file.close()
    outfile.close()
    yfile.close()
    print("total lines = %d" % cnt)
    return


#dataEncoding("./test.out", "enc_test_x.pk", "enc_test_y.pk")
#dataEncoding("./valid.out", "enc_valid_x.pk", "enc_valid_y.pk")
#dataEncoding("./train.out", "enc_train_x.pk", "enc_train_y.pk")

if __name__ == "__main__":
    file = sys.argv[1]
    out_x = "enc_"+file.split(".")[0]+"_x.pk"
    out_y = "enc_"+file.split(".")[0]+"_y.pk"
    print(file, out_x, out_y)
