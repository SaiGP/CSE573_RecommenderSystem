from time import sleep

import numpy as np
from scipy import sparse
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.models import save_model
from keras.preprocessing.sequence import pad_sequences
import _pickle as pickle
from scipy.sparse import csr_matrix
from keras.callbacks import ModelCheckpoint


def Model(dim=(10, 947378)):
    # Input for variable-length sequences of integers
    """inputs = keras.Input(shape=(None,))
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(947379, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.LSTM(64, return_sequences=True, batch_size=1)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])"""
    n = 1
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(20, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(947378, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    checkpoint = ModelCheckpoint("bilstm.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    input("Continue...")
    # train data
    fpx = open("trainx.pk", "rb")
    trainx = pickle.load(fpx)
    fpy = open("y_train.pk", "rb")
    trainy = pickle.load(fpy)
    fpx.close()
    fpy.close()

    # validation data
    vfpx = open("valx.pk", "rb")
    valx = pickle.load(vfpx)
    vfpy = open("y_val.pk", "rb")
    valy = pickle.load(vfpy)
    vfpx.close()
    vfpy.close()

    # test data
    tfpx = open("testx.pk", "rb")
    testx = pickle.load(tfpx)
    tfpy = open("y_test.pk", "rb")
    testy = pickle.load(tfpy)
    tfpx.close()
    tfpy.close()
    ty = np.array(trainy)

    t1x = trainx[:100]
    t1y = trainy[:100]
    v1x = valx[:20]
    v1y = valy[:20]

    print(t1x[0].shape)
    print(t1y[0].shape)
    val = input("press enter...")
    if val == 'n':
        return

    for epoch in range(10):
        for i in range(len(trainx)):
            x = trainx[i].toarray()
            xl, yl = x.shape
            if yl != 947378:
                continue

            # print(x.shape)
            x = np.reshape(x, (1, xl, yl))
            #x = x.T
            y = trainy[i].toarray()
            d1, d2 = y.shape
            y = np.reshape(y, (1, d1, d2))
            #y = y.T
            print(x.shape)
            model.fit(x, y, callbacks=[checkpoint])
            print("next %d" % i)
            # input("$> ")
            if n:
                print("loading weights...")
                model.load_weights("lstm.h5")
                n = 0
        print("fit done...")
        save_model(model, "lstm_1.h5")
        for j in range(len(valx)):
            x = valx[j].toarray()
            xl, yl = x.shape
            if yl != 947378:
                continue

            # print(x.shape)
            x = np.reshape(x, (1, xl, yl))
            #x = x.T
            y = valy[j].toarray()
            d1, d2 = y.shape
            y = np.reshape(y, (1, d1, d2))
            acc = model.evaluate(x, y, batch_size=20)
            print("Epoch %d", epoch)
            print(acc)
        print(model.summary())

    print("DONE!!! ")

    return



def padded_sequence(file, outf):
    fp = open(file, "rb")
    data = pickle.load(fp)
    print(type(data))
    print(type(data[0]))

    for i in range(np.array(data).shape[0]):
        len = data[i].shape[0]
        if len > 10:
            data[i] = data[i][:10,:]
        else:
            x = 10 - len
            data[i] = csr_matrix((data[i].data, data[i].indices, np.pad(data[i].indptr, (x, 0), "edge")))
        print(data[i].shape)

    d = csr_matrix((data[0].data, data[0].indices, np.pad(data[0].indptr, (2, 0), "edge")))
    print(d.shape)
    d = d[:10, :]
    print(d.shape)

    fp = open(outf, "wb")
    pickle.dump(data, fp)


def convert_to_np(file):
    fp = open(file, "rb")
    data = pickle.load(fp)
    out = []
    cnt = 0
    for i in range(len(data)):
        x = data[i].toarray()
        d1, d2 = x.shape
        l = int(947378 - d2)
        if l <= 0:
            out.append(x)
            cnt += 1
        print(x.shape)

    print("total count %d" % cnt)
    output = np.stack(out)
    np.save("train.npy")





# padded_sequence("./Dataprep/encoded data/x_test.pk", "testx.pk")
# padded_sequence("./Dataprep/encoded data/x_train.pk", "trainx.pk")
# padded_sequence("./Dataprep/encoded data/x_val.pk", "valx.pk")

Model()

# convert_to_np("trainx.pk")
