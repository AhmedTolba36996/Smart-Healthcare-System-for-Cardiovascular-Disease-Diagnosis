

# ********************** Import Libraries *******************#

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


from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation# , Dropout


from keras.models import Sequential
from keras.layers import Reshape
from keras.models import Model

from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D , Input
from tensorflow.keras.optimizers import Adam , Adadelta
from keras.callbacks import LearningRateScheduler

import itertools
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

import tensorflow as tf

from flwr.server.strategy import FedAvg
from typing import Callable, Dict, List, Optional, Tuple


# ************* Deep Learning Code *********************** #

    
        # ***********  Building Model ************* #

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
        
        
        # ****************************************#

        # ********** Evaluate Global Model **********#
def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
        # *********** Reading Data ************* #
    
    data=pd.read_csv('mitbih_test.csv' , header=None )
    data.rename(columns={187:"Class"}, inplace=True)
    
    
    X = data.iloc[:,:186].values
    y = data["Class"]
    
    y = to_categorical(y, 5)

    
    # ****************************************#

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(X, y)
        return loss, {"accuracy": accuracy}

    return evaluate

        # ****************************************#



# ******************************************************* #


# ********************** Save Global Model Weights ************************ #




class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights


 # ********************************************************************#
 
strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
		min_available_clients=6, fraction_fit=1.0,
                  min_fit_clients=6 ,eval_fn=get_eval_fn(model)

)
fl.server.start_server(
    "192.168.1.110:25", strategy=strategy, config={"num_rounds": 1})
