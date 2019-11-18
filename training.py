import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical


def main():
    sys.setrecursionlimit(40000)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    df = pd.read_csv('data/training/shutdown_2019-11-17-200500.webm.csv', header=None)
    data = df[0:5].to_numpy()
    IMG_HEIGHT, IMG_WIDTH = data.shape # 5, 57

    data = np.reshape(data, (IMG_HEIGHT, IMG_WIDTH, 1))
    print(IMG_HEIGHT, IMG_WIDTH )

    model = create_model(IMG_HEIGHT, IMG_WIDTH)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    model.summary()
    print(to_categorical([1]).shape)
    print(np.array([data]).shape)

    # model.train_on_batch(np.array([data]), to_categorical([1]))
    # model.train_on_batch(np.array([data]), to_categorical([1]))
    # model.train_on_batch(np.array([data]), to_categorical([1]))

    model.fit(np.array([data]), to_categorical([1]))
    model.fit(np.array([data]), to_categorical([1]))
    model.fit(np.array([data]), to_categorical([1]))


    print(model.predict(np.array([data])))


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
        Dense(2, activation='softmax')
    ])


if __name__ == '__main__':
    main()
