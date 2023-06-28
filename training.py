import numpy as np
from sklearn.model_selection import train_test_split
from tools.models import fcnn
from tools.callbacks import check_point, early_Stopping, lr_schedule
import pandas as pd
import tensorflow as tf


def main():
    dataset = pd.DataFrame()
    model_name = "meomeo"
    path = "./dataset/processed_data"
    with open("dataset/labels/labels.csv") as f:
        gestures = [i.strip() for i in f]
    # print(gestures)
    # gestures = pd.read_csv("./labels/labels.csv", header=None)
    for gesture in gestures:
        dataset = pd.concat((dataset, pd.read_csv(f'{path}/{gesture}.csv')))

    x = dataset.drop("class", axis=1)
    y = dataset["class"]
    random_seed = 42

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, random_state=random_seed, shuffle=True)
    y_train = tf.keras.utils.to_categorical(y_train, 3)
    y_test = tf.keras.utils.to_categorical(y_test, 3)
    print(x.shape, y.shape)
    model = fcnn(21 * 2, 3)
    epochs = 200
    batch_size = 128
    history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, use_multiprocessing=True, shuffle=True,
                        callbacks=[check_point(model_name), lr_schedule()], validation_data=(x_test, y_test))
    np.save(f'./results/history/{model_name}_history.npy', history.history)


if __name__ == "__main__":
    main()
