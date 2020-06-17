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
        print("\n LOADING PICKLED DATASET")

        pickled_images = pickle.load(open((self.data_directory + '/all/pkl_images_all.pkl'), 'rb'))
        pickled_labels = pickle.load(open((self.data_directory + '/all/pkl_labels_all.pkl'), 'rb'))

        _images = np.asarray(pickled_images) / 255.0
        _labels = np.asarray(pickled_labels) / 200

        split_point = int(0.8 * len(_images))

        images, labels = sklearn.utils.shuffle(_images, _labels)

        self.x_train = images[0:split_point]
        self.y_train = labels[0:split_point]

        self.x_test = images[split_point:]
        self.y_test = labels[split_point:]

        self.x_train = np.reshape(np.asarray(self.x_train), (len(self.x_train), 3, 200, 200, 1))
        self.x_test = np.reshape(np.asarray(self.x_test), (len(self.x_test), 3, 200, 200, 1))

        self.y_train = np.reshape(self.y_train, newshape=(len(self.y_train), 6))
        self.y_test = np.reshape(self.y_test, newshape=(len(self.y_test), 6))

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(4)
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(4)

        return train_ds, test_ds


class PositionGenerator:
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds

    def construct_model(self):
        input_1 = tf.keras.Input(shape=(200, 200, 1))
        input_2 = tf.keras.Input(shape=(200, 200, 1))
        input_3 = tf.keras.Input(shape=(200, 200, 1))

        shared_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        z_1 = shared_conv(input_1)
        z_2 = shared_conv(input_2)
        z_3 = shared_conv(input_3)

        concat = tf.keras.layers.Concatenate(axis=3)([z_1, z_2, z_3])

        conv_2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
        conv_2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_2)
        conv_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_2)

        conv_3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_2)
        conv_3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_3)
        conv_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_3)

        conv_4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_3)
        conv_4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_4)
        conv_4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_4)

        flatten = tf.keras.layers.Flatten()(conv_4)
        dense_1 = tf.keras.layers.Dense(units=4096, activation="relu")(flatten)
        dense_2 = tf.keras.layers.Dense(units=6, activation="relu")(dense_1)

        model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=dense_2)

        return model

    def train(self, epochs=200):
        strat = tf.distribute.MirroredStrategy()

        #with strat.scope():
        for i in range(1):
            print("GPUS AVALIBLE ", tf.config.experimental.list_physical_devices('GPU'))
            model = self.construct_model()

            optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

            train_acc_metric = tf.keras.metrics.MeanSquaredError()
            val_acc_metric = tf.keras.metrics.MeanSquaredError()

            @tf.function
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    forward_pass = model([x[:, 0], x[:, 1], x[:, 2]])
                    loss_value = loss_fn(y, forward_pass)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y, forward_pass)

                return loss_value

            @tf.function
            def test_step(x, y):
                val_pass = model([x[:, 0], x[:, 1], x[:, 2]])
                val_acc_metric.update_state(y, val_pass)

            for epoch in range(epochs):
                print("\n\nSTARTING EPOCH {}".format(epoch))

                for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                    loss_value = train_step(x_batch_train, y_batch_train)

                    if step % 100 == 0:
                        print("STEP {} TRAINING LOSS {:.4f}".format(step, loss_value))

                train_acc = train_acc_metric.result()
                print("MEAN SQUARED ERROR OVER EPOCH: {:.4f}".format(train_acc))
                train_acc_metric.reset_states()

                for x_batch_val, y_batch_val in self.test_ds:
                    test_step(x_batch_val, y_batch_val)

                val_acc = val_acc_metric.result()
                val_acc_metric.reset_states()
                print("VALIDATION ERROR: {:.4f}".format(val_acc))

        model.save('saved_posgen')
        print('\nModel saved successfully!')


if __name__ == "__main__":
    train_ds, test_ds = Preprocessing("/Users/micahreich/Documents/VisualEssence/data").train_test_split()
    RG = PositionGenerator(train_ds, test_ds)
    RG.train()
