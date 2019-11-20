#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import os
import glob

from random import randint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,LSTM,Input,Conv1D,MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

#classes = {'Looking':0,
#           'Movingmouse':1,
#           'Righthandnext':2}
#
#t = [a*3-1 for a in range(1,20)]
#
#def read_data():
#    """
#        Takes the list of .csv files, containing the 19 joint positions and creates the keypoints dataframe
#        and labels. For this LSTM model, only the (x,y) locations of each joint are taken.
#        
#        The shape of the final dataframe is in the form of (samples,timesteps,features)
#        where
#        - sample = No of training videos e.g. 12
#        - timesteps = No. of frames in each video (padding added to fit the longest video)
#        - features = No. of joint (x,y) coordinates e.g. 19 joints * 2 coordinates = 38
#        
#        *Returns*: A tuple of the final joint coordinate dataframe and the video labels
#    
#    """ 
#    flist = os.listdir('./data/training-csv/')
#    df = []
#    y = []
#    for file in flist:
#        temp2 = pd.read_csv('./data/training-csv/'+file,header=None)
##        print(temp.shape)
#        temp = np.delete(np.array(temp2),t,axis=1)
#        temp = temp.flatten()
#        temp = np.pad(temp,(0,500*38-len(temp)),mode='constant',constant_values=0)
##        temp = pad_sequences(temp,maxlen=100*57,padding='post')
#        temp = temp.reshape(500,38)
##        print(temp.shape)
#        df.append(temp)
#        
#        label = file.split('_')[0]
#        y.append(classes[label])
#        
#    df2 = np.stack(df,axis=0)
#    y2 = to_categorical(y)
#    return (df2,y2)
#        
#train_x,train_y = read_data()
#
#
#modelname = 'action_lstm'
#model = createModel()
#model.summary()
#
#
#filepath        = modelname + ".hdf5"
#checkpoint      = ModelCheckpoint(filepath, 
#                                  monitor='val_acc', 
#                                  verbose=0, 
#                                  save_best_only=True, 
#                                  mode='max')
#
#                            # Log the epoch detail into csv
#csv_logger      = CSVLogger(modelname +'.csv')
#callbacks_list  = [checkpoint,csv_logger]
#
#model.fit(train_x,train_y,epochs=50,callbacks=callbacks_list)

def createModel():
    """
        Creates a LSTM model with convolutional layers at the start. 
        
        The input shape is given as (timesteps,features) as explained in read_data(). The model is currently set to use
        batch size=1 i.e. SGD type but cna be modified for mini-batch training. 
        
        
    """   
    x= Input(shape=(5,16))
    out = Conv1D(16,3,activation='relu',padding='same')(x)
    out = MaxPooling1D(2)(out)
    out = Conv1D(32,3,activation='relu',padding='same')(out)
    out = MaxPooling1D(2)(out)
    out = LSTM(64,return_sequences=True)(out)
    out = LSTM(32,return_sequences=True)(out)
    out = LSTM(16)(out)
    out = Dense(3,activation='softmax')(out)
    model = Model(x,out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def loadDataFiles():
    return glob.glob('./data/training-csv/*.csv')


def build_classes():
    files = loadDataFiles()
    cs =[]
    classes = {}
    classes_idx = {}
    i = 0

    for f in files:
        cls = os.path.split(f)[1].split('_')[0]
        cs.append(cls)

    cs=sorted(cs)

    for cls in cs:
        if cls not in classes:
            classes[cls] = i
            classes_idx[i] = cls
            i = i + 1
    return classes, classes_idx

def test_sampling():
    files = loadDataFiles()
    datastore = []
    # classes = build_classes()

    for f in files:
        df = pd.read_csv(f, header=None)

    rand_width = 15
    rows = 5

    return sample_data(df, rand_width, rows)

'''
randomly takes a startFrame. Takes 5 rows of data randomly between startFrame and startFrame+rand_width 
'''
def sample_data(df, rand_width, rows):
    startFrame = len(df) - rand_width - 1
    startFrame = randint(0, startFrame)
    rs = []
    for i in range(rows):
        rs.append(randint(startFrame, startFrame + rand_width))
    rs = sorted(rs)

    # print(rs)
    # print(df.loc[rs, :].to_numpy())
    return df.loc[rs, :].to_numpy()

def test():
    print('running test()')
    classes, classes_idx = build_classes()
    print(classes.get('Shutdown'))
    print(classes_idx.get(1))

    build_droplist()


def build_droplist():
    drop_score_i = [a * 3 - 1 for a in range(1, 20)]
    drop_i = [a for a in range(8 * 3, 18 * 3+2)]
    drop_i = drop_i + list(set(drop_score_i) - set(drop_i))
    # drop_i.extend(drop_score_i)
    print(len(drop_i))
    return drop_i

def training():
    sys.setrecursionlimit(40000)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    drop_score_i = build_droplist()

    ACTION_ROWS, ACTION_WIDTH = 5, 57-len(drop_score_i)  # 5, 57, drop 41

    print(ACTION_ROWS, ACTION_WIDTH)
    classes, _ = build_classes()

    # drop_score_i = [a * 3 - 1 for a in range(1, 20)]
    # drop_i = [a for a in range(8*3, 13*3+2)]
    # drop_i = drop_i.append(drop_score_i)

    model = createModel()
#    print('len(classes):',classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    modelname = 'action_lstm'

    filepath = modelname + ".hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    # Log the epoch detail into csv
    csv_logger = CSVLogger(modelname + '.csv')
    callbacks_list = [checkpoint, csv_logger]


    files = loadDataFiles()
    datastore = []

    # build datastore from csv files
    for f in files:
        df = pd.read_csv(f, header=None)
        cls = os.path.split(f)[1].split('_')[0]
        datastore.append([f, cls, df])


    # train 50 ecpochs, resampling training data.
    for j in range(50): # change to low single digits to see underfit error.
        # train 1 epoch
        datas = []
        ys = []
        for d in datastore:
            for i in range(10):
                data = sample_data(d[2], 10, 5)
                # print('before delete:',data.shape)
                data = np.delete(np.array(data), drop_score_i, axis=1)
                data = np.reshape(data, (ACTION_ROWS, ACTION_WIDTH))
                # print('after delete:',data.shape)

                datas.append(data)
                cls = d[1]
                ys.append(classes.get(cls))

        # print(to_categorical(ys).shape)
        # print(np.array(datas).shape)

        model.fit(np.array(datas), to_categorical(ys), callbacks=callbacks_list)


    tests = []
    yts = []
    for d in datastore:

        for i in range(1):
            # data = np.reshape(sample_data(d[2], 20, 5), (ACTION_ROWS, ACTION_WIDTH, 1))
            data = sample_data(d[2], 20, 5)
            data = np.delete(np.array(data), drop_score_i, axis=1)
            data = np.reshape(data, (ACTION_ROWS, ACTION_WIDTH))
            tests.append(data)
            cls = d[1]
            yts.append(classes.get(cls))

        # print(to_categorical(ys).shape)
        # print(np.array(datas).shape)

    preds= model.predict(np.array(tests))
    print('Pred:',np.argmax(preds, axis=1))
    print('Test labels:',yts)

    print(confusion_matrix(yts, np.argmax(preds, axis=1)))

def main():
    training()
    # test_sampling()
    # test()

if __name__ == '__main__':
    main()
