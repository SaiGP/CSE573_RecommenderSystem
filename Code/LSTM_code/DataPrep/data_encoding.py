import csv
import json
import os.path
import sys
from time import sleep

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from random import randint
import _pickle as cPickle
from sklearn.model_selection import train_test_split



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


def data_split():
    fp = open("train_seq.out", "r")
    train_data =[]
    d = fp.readline()
    while d:
        data = d.split('\t')
        # print(data)
        train_data.append(data)
        d = fp.readline()
    for i in range(5):
        print("train_data %d" % i)
        train_data, d1 = train_test_split(train_data, test_size=0.2)
        outfile = "train_seq" + str(i+1) +".out"
        print("split and write to " + outfile)
        outfp = open(outfile, "w")
        for td in d1:
            for t in td:
                outfp.write(str(t) + "\t")
        d1.clear()
        outfp.close()

    fp.close()
    return


#data_split()

#dataEncoding("./test.out", "enc_test_x.pk", "enc_test_y.pk")
#dataEncoding("./valid.out", "enc_valid_x.pk", "enc_valid_y.pk")
#dataEncoding("./train.out", "enc_train_x.pk", "enc_train_y.pk")


def labEncoding(infile, outfile):
    infp = open(infile, "r")
    line = infp.readline()

    # encoder fit
    item_ids = np.loadtxt("uiids.out", delimiter=',', dtype='str')
    uiids = np.unique(item_ids)
    labenc = LabelEncoder()
    labenc.fit(uiids)
    outx = outfile + "_x.np"
    outxfp = open(outx, "a")

    outy = outfile + "_y.np"
    outyfp = open(outy, "a")
    x_val = []
    y_val = []
    print("Encoding ...")
    while line:
        data = line.split("\t")
        x = data[4].split(",")
        x_lab = labenc.transform(x)
        y = [data[1]]
        y_lab = labenc.transform(y)
        x_val = np.array([x_lab, data[5].split(",")])
        y_val= np.array([y_lab, [data[2]]])
        print("X: ")
        print(x_val)
        print("Y: ")
        print(y_val)
        np.savetxt(outxfp, x_val, delimiter=",", newline='\n', fmt='%s')
        np.savetxt(outyfp, y_val, delimiter=",", newline='\n', fmt='%s')
        line = infp.readline()
    print("saved encoding...")

    #np.save(outx, np.array(x_val))

    #np.save(outy, np.array(y_val))
    outxfp.close()
    outyfp.close()
    infp.close()
    return

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    labEncoding(infile, outfile)
