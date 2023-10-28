import os 
import pandas as pd
import numpy as np

import pandas as pd
from Bio import SeqIO
import os
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Input, Flatten, LSTM, Dropout, Bidirectional, LeakyReLU, Reshape, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

# performance matrices
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score

# plots
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy import array
from numpy import argmax

from tensorflow.keras.regularizers import l1, l2


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import imblearn
a = random.sample(range(1, 1000000), 100)

os.chdir("/home/t326h379/OGP_30_Percent_CD_HIT_1185")

df_training_positive = pd.read_csv('ESM2_2B_Feature_Extraction_O_linked_Training_Positive_4885_Sites_less_friday_subash.txt',header=None)

df_training_negative = pd.read_csv('ESM2_3B_Feature_Extraction_O_linked_Training_Negative_114307_Sites_less_Friday_Subash.txt',header=None)

df_independent_negative = pd.read_csv('ESM2_3B_Feature_Extraction_O_linked_Testing_Negative_11466_Sites_less.txt',header=None)

df_indpendent_positive = pd.read_csv('ESM2_3B_Feature_Extraction_O_linked_Testing_Positive_375_Sites_less_Friday_Subash.txt',header=None)

for i in range(len(a)):
    
    seed = a[i]
    Header_name = ["Position","PID","Position_redundant","81 Window sequence","S or T"]

    col_of_feature = [i for i in range(1,2561)]

    Header_name = Header_name + col_of_feature

    df_training_positive.columns = Header_name
    df_training_negative.columns = Header_name

    frames = [df_training_positive, df_training_negative]

    O_linked_training = pd.concat(frames,ignore_index = True)

    df_Train_array = O_linked_training.drop(["Position","PID","Position_redundant","81 Window sequence","S or T"],axis=1)
    df_Train_array = np.array(df_Train_array)

    X_train_full = df_Train_array

    y_train_full = np.array([1]*3783+[0]*85553)

    from sklearn.metrics import roc_curve, roc_auc_score, classification_report, auc
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state = seed)
    X_train, y_train = rus.fit_resample(X_train_full,y_train_full)

    x_train_train, x_val_val, y_train_train, y_val_val = train_test_split(X_train, y_train, test_size=0.1)  

    y_train_train = tf.keras.utils.to_categorical(y_train_train,2)
    y_val_val = tf.keras.utils.to_categorical(y_val_val,2)


    model = Sequential()

    model.add(Input(shape=(2560,)))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(256,activation='relu',name="Dense_1"))
    model.add(Dropout(0.3))

    model.add(Dense(32,activation='relu',name="Dense_2"))
    model.add(Dropout(0.3))

    model.add(Dense(2,activation='softmax',name="Dense_3"))


    model.compile(optimizer=tf.keras.optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="ROC_ROC_Premise_Assumption.h5", 
                                    monitor = 'val_accuracy',
                                    verbose=0, 
                                    save_weights_only=False,
                                    save_best_only=True)

    reduce_lr_acc = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.001, patience=5, verbose=1, min_delta=1e-4, mode='max')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,mode='max')

    history = model.fit(x_train_train, y_train_train,epochs=400,verbose=1,batch_size=256,
                            callbacks=[checkpointer,reduce_lr_acc, early_stopping],validation_data=(x_val_val, y_val_val))

    Header_name = ["Position","PID","Position_redundant","81 Window sequence","S or T"]

    col_of_feature = [i for i in range(1,2561)]

    Header_name = Header_name + col_of_feature

    df_indpendent_positive.columns = Header_name
    df_independent_negative.columns = Header_name

    frames = [df_indpendent_positive, df_independent_negative]

    O_linked_test = pd.concat(frames,ignore_index = True)

    df_Test_array = O_linked_test.drop(["Position","PID","Position_redundant","81 Window sequence","S or T"],axis=1)
    df_Test_array = np.array(df_Test_array)

    X_test_test_full = df_Test_array

    y_test_test_full = np.array([1]*322+[0]*8668)

    from sklearn.metrics import roc_curve, roc_auc_score, classification_report, auc
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state = seed)
    X_Independent, y_independent = rus.fit_resample(X_test_test_full,y_test_test_full)

    Y_pred = model.predict(X_Independent)
    Y_pred = (Y_pred > 0.5)
    y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]
    y_pred = np.array(y_pred)

    confusion = confusion_matrix(y_independent,y_pred)

    print("Matthews Correlation : ",matthews_corrcoef(y_independent, y_pred))
    print("Confusion Matrix : \n",confusion_matrix(y_independent, y_pred))
    print("Accuracy on test set:   ",accuracy_score(y_independent, y_pred))

    cm = confusion_matrix(y_independent, y_pred)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    mcc = matthews_corrcoef(y_independent, y_pred)

    Sensitivity = TP/(TP+FN)

    Specificity = TN/(TN+FP)

    print("Sensitivity:   ",Sensitivity,"\t","Specificity:   ",Specificity)

    print(classification_report(y_independent, y_pred))

    fpr, tpr, _ = roc_curve(y_independent, y_pred)

    roc_auc_test = auc(fpr,tpr)



    print("Area Under Curve:   ",roc_auc_test)

    print("Precision:   ",TP/(TP+FP))