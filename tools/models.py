import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Input, layers, activations
from keras.layers import Dense, BatchNormalization, Dropout
from keras.activations import relu, softmax


def fcnn(INPUT_DIM, OUTPUT_DIM, write_summary=False):
    model = keras.Sequential([
        Input((INPUT_DIM, )),
        Dropout(0.2),

        Dense(32, activation="relu"),
        BatchNormalization(),
        # relu(),
        Dropout(0.4),

        Dense(16, activation='relu'),
        BatchNormalization(),
        # relu(),

    Dense(10, activation='relu'),

        Dense(OUTPUT_DIM, activation="softmax"),
        # softmax(),
    ])
    if write_summary:
        print(model.summary())
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

