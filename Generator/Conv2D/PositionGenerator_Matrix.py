import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K


class PositionGenerator:
    def __init__(self, data_directory, x_train, y_train, x_test, y_test):
        self.data_directory = data_directory
        self.x_train = x_train / 255.0
        self.y_train = y_train

        self.x_test = x_test / 255.0
        self.y_test = y_test

    def train_test_split(self):
        self.x_train = np.reshape(self.x_train, newshape=(len(self.x_train), 3, 40000))
        self.x_test = np.reshape(self.x_test, newshape=(len(self.x_test), 3, 40000))

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(64)
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(64)

        return train_ds, test_ds

    def construct_model(self):
        model = Sequential()
        model.add(Dense(3, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(6, activation="relu"))

    def train(self):
        pass


if __name__ == "__main__":
    RG = PositionGenerator("test")
