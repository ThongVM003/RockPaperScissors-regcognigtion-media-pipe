from keras import Sequential, Input
from keras.layers import Dense, BatchNormalization, Dropout


def fcnn(INPUT_DIM, OUTPUT_DIM, write_summary=False):
    model = Sequential(
        [
            Input((INPUT_DIM,)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(16, activation="relu"),
            BatchNormalization(),
            Dense(10, activation="relu"),
            Dense(OUTPUT_DIM, activation="softmax"),
        ]
    )
    if write_summary:
        print(model.summary())
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
