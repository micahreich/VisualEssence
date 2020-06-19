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

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(len(self.x_train))
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).shuffle(len(self.x_test))

        return train_ds, test_ds


class PositionGenerator:
    def __init__(self, train_ds, test_ds):
        self.BATCH_SIZE = 64
        self.EPOCHS = 12

        self.train_ds = train_ds.batch(self.BATCH_SIZE)
        self.test_ds = test_ds.batch(self.BATCH_SIZE)

    def conv_block(self, input, kernel_size, filters, bn_axis=-1, strides=(2, 2, 2)):
        filter_1, filter_2, filter_3 = filters
        kernel_size_1, kernel_size_2, kernel_size_3 = kernel_size

        x = tf.keras.layers.Conv3D(filter_1, kernel_size_1, padding='same', strides=strides)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_2, kernel_size_2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_3, kernel_size_3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        z = tf.keras.layers.Conv3D(filter_3, kernel_size_3, padding='same', strides=strides)(input)
        z = tf.keras.layers.BatchNormalization()(z)

        x = tf.keras.layers.add([x, z])
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    def identity_block(self, input, kernel_size, filters, bn_axis=-1, strides=(2, 2, 2)):
        filter_1, filter_2, filter_3 = filters
        kernel_size_1, kernel_size_2, kernel_size_3 = kernel_size

        x = tf.keras.layers.Conv3D(filter_1, kernel_size_1, padding='same')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_2, kernel_size_2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_3, kernel_size_3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    def ResNet3D(self, bn_axis=-1):
        stacked_input = tf.keras.layers.Input(shape=(1, 224, 224, 3))

        # Input Conv Block
        x = tf.keras.layers.ZeroPadding3D(padding=(3, 3, 3))(stacked_input)
        x = tf.keras.layers.Conv3D(64,
                                   (7, 7, 7),
                                   strides=(2, 2, 2),
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1))(x)
        x = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(x)

        # Conv Block 1
        x = self.conv_block(x, [1, 3, 1], [64, 64, 256])

        # Identity x 2
        x = self.identity_block(x, [1, 3, 1], [64, 64, 256])
        x = self.identity_block(x, [1, 3, 1], [64, 64, 256])

        # Conv Block 2
        x = self.conv_block(x, [1, 3, 1], [128, 128, 512])

        # Identity x 3
        x = self.identity_block(x, [1, 3, 1], [128, 128, 512])
        x = self.identity_block(x, [1, 3, 1], [128, 128, 512])
        x = self.identity_block(x, [1, 3, 1], [128, 128, 512])

        # Conv Block 3
        x = self.conv_block(x, [1, 3, 1], [256, 256, 1024])

        # Identity x 5
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])

        # Conv Block 4
        x = self.conv_block(x, [1, 3, 1], [512, 512, 2048])

        # Identity x 2
        x = self.identity_block(x, [1, 3, 1], [512, 512, 2048])
        x = self.identity_block(x, [1, 3, 1], [512, 512, 2048])

        # Output
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Dense(units=6, activation='relu')(x)

        model = tf.keras.models.Model(stacked_input, x)

        return model

    def train(self):
        strat = tf.distribute.MirroredStrategy()
        print("GPUS AVALIBLE ", tf.config.experimental.list_physical_devices('GPU'))

        with strat.scope():
            model = self.ResNet3D()

            model.compile(optimizer=tf.keras.optimizers.Adagrad(),
                          loss=tf.keras.losses.MeanSquaredError(reduction="auto"),
                          metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CosineSimilarity()])

        print('\nBEGINNING MODEL TRAINING')
        model.fit(self.train_ds, epochs=self.EPOCHS, validation_data=self.test_ds)

        print('\nBEGINNING MODEL VALIDATION')
        model.evaluate(self.test_ds)

        model.save('saved_pos_gen')
        print('\nMODEL SAVED SUCCESSFULLY!')


if __name__ == "__main__":
    #train_ds, test_ds = Preprocessing("/Users/micahreich/Documents/VisualEssence/data").train_test_split()
    RG = PositionGenerator().ResNet3D()
    #RG.train()
