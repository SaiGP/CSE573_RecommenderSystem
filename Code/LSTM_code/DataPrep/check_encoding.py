import _pickle as cPickle
import os
import pickle
import random

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import csv
from keras.preprocessing.sequence import pad_sequences
import tensorflow


def check():
    file = "./enc_valid1_y.pk"
    data = []
    fp = open(file, "rb")
    xpk = cPickle.Unpickler(fp)

    while True:
        try:
            data.append(xpk.load())
        except EOFError:
            break

    print(data)

    with open("./fit_seq.out", "r") as fsfp:
        fitread = csv.reader(fsfp)
        fit_seq = []
        for row in fitread:
            fit_seq.append(row)

    print(fit_seq)

    ohe_item = OneHotEncoder()
    ohe_item.fit(fit_seq)

    outd = ohe_item.inverse_transform(data[0])
    print(outd)
    d = [[0,0]]



def combine():
    files = os.listdir("./")
    data_x = []

    for file in files:
        if '_y.pk' in file:
            print(file)
            with open(file, "rb") as fp:
                xpk = cPickle.Unpickler(fp)
                while True:
                    try:
                        data_x.append(xpk.load())
                    except EOFError:
                        break

                fp.close()

    ofp = open("ydata.pk", "wb")
    pickle.dump(data_x, ofp)
    return

# combine()

def split():
    file = "xdata.pk"
    xfp = open(file, "rb")
    xdata = pickle.load(xfp)
    yfp = open("ydata.pk", "rb")
    ydata = pickle.load(yfp)

    print(len(xdata), len(ydata))
    xdata = xdata[:len(ydata)]
    print(len(xdata), len(ydata))

    arr = list(range(len(ydata)))
    print(type(arr))

    ind = random.sample(range(len(arr)), int(len(arr)/10))

    arr_test = [arr[i] for i in ind]
    xa = list(set(arr) - set(arr_test))

    ind = random.sample(range(len(xa)), int(len(arr)/10))
    arr_val = [xa[i] for i in ind]
    arr_train = list(set(xa) - set(arr_val))

    x_train = [xdata[a] for a in arr_train]
    x_test = [xdata[a] for a in arr_test]
    x_val = [xdata[a] for a in arr_val]

    y_train = [ydata[a] for a in arr_train]
    y_test = [ydata[a] for a in arr_test]
    y_val = [ydata[a] for a in arr_val]


    xtfp = open("x_train.pk", "wb")
    xtestfp = open("x_test.pk", "wb")
    xvfp = open("x_val.pk", "wb")

    ytfp = open("y_train.pk", "wb")
    ytestfp = open("y_test.pk", "wb")
    yvfp = open("y_val.pk", "wb")

    print("dump...")
    pickle.dump(x_train, xtfp)
    pickle.dump(x_test, xtestfp)
    pickle.dump(x_val, xvfp)
    pickle.dump(y_train, ytfp)
    pickle.dump(y_test, ytestfp)
    pickle.dump(y_val, yvfp)

    xtfp.close()
    xtestfp.close()
    xvfp.close()

    ytfp.close()
    ytestfp.close()
    yvfp.close()


split()
