#******************** Import Libraries ************************#

import flwr as fl
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from sklearn.utils import class_weight
import warnings


from keras.models import Sequential
from keras.layers import Reshape
from keras.models import Model

from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D , Input
from tensorflow.keras.optimizers import Adam , Adadelta
from keras.callbacks import LearningRateScheduler

import itertools
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

import tensorflow as tf

from imblearn.over_sampling import SMOTE


warnings.filterwarnings('ignore')

#*********************************************************#
#*********************************************************#


df_train=pd.read_csv('mitbih_train.csv' , header=None)
df_test=pd.read_csv('mitbih_test.csv' , header=None )


all_data = [df_train, df_test]
data = pd.concat(all_data)

data.rename(columns={187:"Class"}, inplace=True)
mapping = {     0. : 'Normal Beat',
               1. : 'Supraventricular premature beat',
               2. : 'Premature ventricular contraction',
               3. : 'Fusion of ventricular',
               4. : 'Unclassifiable beat'}

data['label'] = data.iloc[:, -1].map(mapping)


# ************ Smot Data ******************************

df_0=(data[data['Class']==0]).sample(n=7000,random_state=42 , replace=True)
df_1=(data[data['Class']==1]).sample(n=7000,random_state=42 , replace=True)
df_2=(data[data['Class']==2]).sample(n=7000,random_state=42 , replace=True)
df_3=data[data['Class']==3]
df_4=data[data['Class']==4]


data=pd.concat([df_0,df_1,df_2,df_3,df_4])

X = data.iloc[:,:186].values
y = data["Class"]

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)



from tensorflow.keras.utils import to_categorical
y = to_categorical(y, 5)



# ******  split data into training and test set ********************
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)



#************************ Building Model *************************#

from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation# , Dropout



inp = Input(shape=(186, 1))
C = Conv1D(filters=64, kernel_size=5, strides=1)(inp)

C11 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(C)
A11 = Activation("relu")(C11)
C12 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(A11)
S11 = Add()([C12, C])
A12 = Activation("relu")(S11)
M11 = MaxPooling1D(pool_size=5, strides=2)(A12)


C21 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(M11)
A21 = Activation("relu")(C21)
C22 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(A21)
S21 = Add()([C22, M11])
A22 = Activation("relu")(S11)
M21 = MaxPooling1D(pool_size=5, strides=2)(A22)


C31 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(M21)
A31 = Activation("relu")(C31)
C32 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(A31)
S31 = Add()([C32, M21])
A32 = Activation("relu")(S31)
M31 = MaxPooling1D(pool_size=5, strides=2)(A32)


C41 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(M31)
A41 = Activation("relu")(C41)
C42 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(A41)
S41 = Add()([C42, M31])
A42 = Activation("relu")(S41)
M41 = MaxPooling1D(pool_size=5, strides=2)(A42)


C51 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(M41)
A51 = Activation("relu")(C51)
C52 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(A51)
S51 = Add()([C52, M41])
A52 = Activation("relu")(S51)
M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

F1 = Flatten()(M51)

D1 = Dense(32)(F1)
A6 = Activation("relu")(D1)
D2 = Dense(32)(A6)
D3 = Dense(5)(D2)
A7 = Softmax()(D3)

model = Model(inputs=inp, outputs=A7)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#******************************************************************#
#******************************************************************#


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}



fl.client.start_numpy_client("192.168.1.110:25", client=CifarClient())






