
import numpy as np
from scipy import sparse
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.models import load_model
from keras.models import save_model
from keras.preprocessing.sequence import pad_sequences
import _pickle as pickle
from scipy.sparse import csr_matrix
from keras.callbacks import ModelCheckpoint


def Model(dim=(10, 947378)):

    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(20, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(947378, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint("bilstm.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')

    model = load_model("lstm_1.h5")
    print(model.summary())
    p = input("$>")
    if p == "n":
        return

    # test data
    tfpx = open("testx.pk", "rb")
    testx = pickle.load(tfpx)
    tfpy = open("y_test.pk", "rb")
    testy = pickle.load(tfpy)
    tfpx.close()
    tfpy.close()
    ty = np.array(testy)

    for j in range(len(testx)):
            x = testx[j].toarray()
            xl, yl = x.shape
            if yl != 947378:
                continue

            # print(x.shape)
            x = np.reshape(x, (1, xl, yl))
            #x = x.T
            y = testy[j].toarray()
            d1, d2 = y.shape
            y = np.reshape(y, (1, d1, d2))
            acc = model.evaluate(x, y, batch_size=20)
            print(acc)

    model.summary()

Model()
