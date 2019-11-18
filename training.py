import pandas as pd
import numpy as np
import sys
import os
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from random import randint
from sklearn.metrics import confusion_matrix



def training():
    sys.setrecursionlimit(40000)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    IMG_HEIGHT, IMG_WIDTH = 5, 57  # 5, 57

    print(IMG_HEIGHT, IMG_WIDTH)

    model = create_model(IMG_HEIGHT, IMG_WIDTH)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    files = loadDataFiles()
    datastore = []
    classes = build_classes()

    for f in files:
        df = pd.read_csv(f, header=None)
        cls = os.path.split(f)[1].split('_')[0]
        datastore.append([f, cls, df])


    for j in range(30): # change to low single digits to see underfit error.
        # train 1 epoch
        datas = []
        ys = []
        for d in datastore:
            for i in range(10):
                data = np.reshape(sample_data(d[2], 20, 5), (IMG_HEIGHT, IMG_WIDTH, 1))
                datas.append(data)
                cls = d[1]
                ys.append(classes.get(cls))

        # print(to_categorical(ys).shape)
        # print(np.array(datas).shape)

        model.fit(np.array(datas), to_categorical(ys))


    tests = []
    yts = []
    for d in datastore:

        for i in range(1):
            data = np.reshape(sample_data(d[2], 20, 5), (IMG_HEIGHT, IMG_WIDTH, 1))
            tests.append(data)
            cls = d[1]
            yts.append(classes.get(cls))

        # print(to_categorical(ys).shape)
        # print(np.array(datas).shape)

    preds= model.predict(np.array(tests))
    print(np.argmax(preds, axis=1))
    print(yts)

    print(confusion_matrix(yts, np.argmax(preds, axis=1)))



def create_model(IMG_HEIGHT, IMG_WIDTH):
    return Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        # MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        # MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        # MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(4, activation='softmax')
    ])


def loadDataFiles():
    return glob.glob('data/training-csv/*.csv')


def build_classes():
    files = loadDataFiles()
    cs =[]
    classes = {}
    i = 0

    for f in files:
        cls = os.path.split(f)[1].split('_')[0]
        cs.append(cls)

    cs=sorted(cs)

    for cls in cs:
        if cls not in classes:
            classes[cls] = i
            i = i + 1
    return classes

def test_sampling():
    files = loadDataFiles()
    datastore = []
    classes = build_classes()

    for f in files:
        df = pd.read_csv(f, header=None)

    rand_width = 15
    rows = 5

    return sample_data(df, rand_width, rows)

'''
takes 5 rows of data randomly within rand_width 
'''
def sample_data(df, rand_width, rows):
    base = len(df) - rand_width - 1
    base = randint(0, base)
    rs = []
    for i in range(rows):
        rs.append(randint(base, base + rand_width))
    rs = sorted(rs)

    # print(rs)
    # print(df.loc[rs, :].to_numpy())
    return df.loc[rs, :].to_numpy()

def test():
    classes = build_classes()
    print(classes.get('shutdown'))

def main():
    training()
    # test_sampling()
    # test()

if __name__ == '__main__':
    main()
