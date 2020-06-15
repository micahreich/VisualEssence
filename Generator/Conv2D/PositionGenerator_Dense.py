import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K


class Preprocessing:
    def __init__(self, data_directory, x_train, y_train, x_test, y_test):
        self.data_directory = data_directory
        self.x_train = x_train / 255.0  # (NUM SAMPLES, 3, 200, 200, 1)
        self.y_train = y_train  # (NUM SAMPLES, 3, 2)

        self.x_test = x_test / 255.0  # (NUM SAMPLES, 3, 200, 200, 1)
        self.y_test = y_test  # (NUM SAMPLES, 3, 2)

    def train_test_split(self):
        self.x_train = np.reshape(self.x_train, newshape=(-1, 3, 200, 200, 1))
        self.x_test = np.reshape(self.x_train, newshape=(-1, 3, 200, 200, 1))

        self.y_train = np.reshape(self.x_train, newshape=(-1, 6))
        self.y_test = np.reshape(self.x_train, newshape=(-1, 6))

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(64)
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(64)

        return train_ds, test_ds


class PositionGenerator:
    """def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds"""

    def construct_model(self):
        input_1 = tf.keras.Input(shape=(200, 200, 1))
        input_2 = tf.keras.Input(shape=(200, 200, 1))
        input_3 = tf.keras.Input(shape=(200, 200, 1))

        shared_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        z_1 = shared_conv(input_1)
        z_2 = shared_conv(input_2)
        z_3 = shared_conv(input_3)

        concat = tf.keras.layers.Concatenate(axis=3)([z_1, z_2, z_3])

        conv_2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(concat)
        conv_2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
        conv_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_2)

        conv_3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
        conv_3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
        conv_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_3)

        conv_4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
        conv_4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
        conv_4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)

        flatten = tf.keras.layers.Flatten()(conv_4)
        dense_1 = tf.keras.layers.Dense(units=4096, activation="relu")(flatten)
        dense_2 = tf.keras.layers.Dense(units=6, activation="relu")(dense_1)

        model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=dense_2)
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return model

    def train(self):
        pass


if __name__ == "__main__":
    RG = PositionGenerator()
    RG.construct_model()
