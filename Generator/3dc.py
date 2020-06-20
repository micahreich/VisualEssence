import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import pickle
import sklearn


class Preprocessing:
    def __init__(self, data_directory):
        self.data_directory = data_directory

        self.x_train = None
        self.y_train = None
	
        self.x_test = None
        self.y_test = None

    def train_test_split(self):
        print("\nLOADING PICKLED DATASET")

        pickled_images = pickle.load(open((self.data_directory + '/all/pkl_images_all.pkl'), 'rb'))
        pickled_labels = pickle.load(open((self.data_directory + '/all/pkl_labels_all.pkl'), 'rb'))

        stacked_images = []

        for triplet in pickled_images:
            stacked_images.append(np.dstack((triplet[0], triplet[1], triplet[2])))

        images = np.asarray(stacked_images) / 255.0
        labels = np.asarray(pickled_labels) / 200

        split_point = int(0.8 * len(images))

        self.x_train = images[0:split_point]
        self.y_train = labels[0:split_point]

        self.x_test = images[split_point:]
        self.y_test = labels[split_point:]

        self.x_train = np.reshape(np.asarray(self.x_train), (len(self.x_train), 1, 200, 200, 3))
        self.x_test = np.reshape(np.asarray(self.x_test), (len(self.x_test), 1, 200, 200, 3))

        self.y_train = np.reshape(self.y_train, newshape=(len(self.y_train), 6))
        self.y_test = np.reshape(self.y_test, newshape=(len(self.y_test), 6))

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(len(self.x_train)).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).shuffle(len(self.x_test)).batch(32)

        return train_ds, test_ds


class PositionGenerator:
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds

    def construct_model(self):
        input = tf.keras.layers.Input(shape=(1, 200, 200, 3))

        conv_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same', data_format='channels_last', activation='relu')(input)
        conv_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same', data_format='channels_last', activation='relu')(conv_1)
        max_pool_1 = tf.keras.layers.MaxPool3D(pool_size=2, padding='same', data_format='channels_last', strides=2)(conv_1)

        conv_2 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same', data_format='channels_last', activation='relu')(max_pool_1)
        conv_2 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same', data_format='channels_last', activation='relu')(conv_2)
        max_pool_2 = tf.keras.layers.MaxPool3D(pool_size=2, strides=2, padding='same', data_format='channels_last',)(conv_2)

        conv_3 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=2, padding='same', data_format='channels_last', activation='relu')(max_pool_2)
        conv_3 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=2, padding='same', data_format='channels_last', activation='relu')(conv_3)
        conv_3 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=2, padding='same', data_format='channels_last', activation='relu')(conv_3)

        flatten = tf.keras.layers.Flatten()(conv_3)
        dense_1 = tf.keras.layers.Dense(units=4096, activation="relu")(flatten)
        dense_2 = tf.keras.layers.Dense(units=6, activation="relu")(dense_1)

        model = tf.keras.Model(inputs=input, outputs=dense_2)

        return model

    def train(self, epochs=20):
        strat = tf.distribute.MirroredStrategy()
        print("GPUS AVALIBLE ", tf.config.experimental.list_physical_devices('GPU'))

        with strat.scope():
            model = self.construct_model()

            model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001),
                          loss=tf.keras.losses.MeanSquaredError(reduction="auto"),
                          metrics=['accuracy'])

        print('\nBEGINNING MODEL TRAINING')
        model.fit(self.train_ds, epochs=epochs, validation_data=self.test_ds)

        print('\nBEGINNING MODEL VALIDATION')
        model.evaluate(self.test_ds)

        model.save('saved_pos_gen')
        print('\nMODEL SAVED SUCCESSFULLY!')


if __name__ == "__main__":
    train_ds, test_ds = Preprocessing("/Users/micahreich/Documents/VisualEssence/data").train_test_split()
    RG = PositionGenerator(train_ds, test_ds)
    RG.train()

