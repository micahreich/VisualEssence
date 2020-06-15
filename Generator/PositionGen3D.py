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
        input = tf.keras.layers.Input(shape=(3, 200, 200, 1))

        # Conv Block 1 (Conv3D, Conv3D, MaxPool3D)
        conv_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                        activation='relu')(input)
        conv_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                        activation='relu')(conv_1)
        maxpool_1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_1)

        # Conv Block 2 (Conv3D, Conv3D, MaxPool3D)
        conv_2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                        activation='relu')(maxpool_1)
        conv_2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                        activation='relu')(conv_2)
        #maxpool_2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_2)

        # Conv Block 3 (Conv3D, Conv3D, MaxPool3D)
        conv_3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=1, padding='same',
                                        activation='relu')(conv_2)
        conv_3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=2, padding='same',
                                        activation='relu')(conv_3)
        #maxpool_3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_3)

        # Conv Block 4 (Conv3D, Conv3D, MaxPool3D)
        conv_4 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=2, padding='same',
                                        activation='relu')(conv_3)
        conv_4 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=2, padding='same',
                                        activation='relu')(conv_4)
        #maxpool_4 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_4)

        # Output layers
        flatten = tf.keras.layers.Flatten()(conv_4)
        fc_1 = tf.keras.layers.Dense(units=4096, activation='relu')(flatten)
        fc_2 = tf.keras.layers.Dense(units=6, activation='relu')(fc_1)

        model = tf.keras.Model(inputs=input, outputs=fc_2)
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return model

    def train(self):
        pass


if __name__ == "__main__":
    RG = PositionGenerator()
    RG.construct_model()
