import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import DiscriminatorDataGen
import numpy as np
from PIL import Image


class IconDiscriminator:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train/255.0
        self.y_train = to_categorical(y_train)

        self.x_test = x_test/255.0
        self.y_test = to_categorical(y_test)

    def train_test_split(self):
        self.x_train = np.reshape(self.x_train, newshape=(len(self.x_train), 200, 200, 1))
        self.x_test = np.reshape(self.x_test, newshape=(len(self.x_test), 200, 200, 1))

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(64)

        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(64)

        return train_ds, test_ds

    def construct_model(self):
        model = Sequential()
        # Convolution Block 1 (Conv2D, Conv2D, MaxPool2D)
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same",
                         input_shape=(self.x_train.shape[1], self.x_train.shape[1], self.x_train.shape[3])))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Convolution Block 2 (Conv2D, Conv2D, MaxPool2D)
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Convolution Block 3 (Conv2D, Conv2D, Conv2D, MaxPool2D)
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Convolution Block 4 (Conv2D, Conv2D, Conv2D,MaxPool2D)
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Convolution Block 5 (Conv2D, Conv2D, Conv2D, MaxPool2D)
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))

        # Dense Block (Dense, Dense)
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))

        # Softmax Block (Softmax Output)
        model.add(Dense(units=2, activation="softmax"))

        return model

    def train(self, epochs=20):
        strat = tf.distribute.MirroredStrategy()

        with strat.scope():
            print("GPUS AVALIBLE ", tf.config.experimental.list_physical_devices('GPU'))

            train_ds, test_ds = self.train_test_split()

            model = self.construct_model()

            model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001),
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy'])

            print('\nBeginning model training')

            model.fit(train_ds, epochs=epochs)

            print('\nBeginning model validation')

            model.evaluate(test_ds)
            # blah
            model.save('saved_discriminator')
            print('\nModel saved successfully!')

    def test(self, path_to_image):
        model = tf.keras.models.load_model('saved_discriminator')
        image_arr = (255 - np.asarray(Image.open(path_to_image)))[:, :, 3]

        return model.predict(image_arr/255.0)


if __name__ == "__main__":
    # Generate training, testing datasets
    DS = DiscriminatorDataGen.DatasetGenerator(60000, "/nethome/mreich8/VisualEssence/data/cnn_data_backup/cnn_data", "LOAD_PICKLE")
    data = DS.generate_dataset()

    CNN = IconDiscriminator(data[0], data[1], data[2], data[3])
    CNN.train()

