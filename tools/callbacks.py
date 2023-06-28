import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def check_point(name='meomeo'):
    return ModelCheckpoint(filepath=f'results/trained_models/{name}.hdf5',
                           monitor='loss',
                           verbose=0,
                           save_best_only=True,
                           mode='auto')




def early_Stopping(pat=3):
    return EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=pat,
        verbose=0,
        mode="auto",
        restore_best_weights=False,
    )


def lr_schedule():
    return ReduceLROnPlateau(monitor="loss")