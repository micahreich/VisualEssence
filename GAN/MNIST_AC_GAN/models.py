import tensorflow as tf
from tensorflow.keras.layers import *


class AC_GAN:
    def __init__(self):
        self.img_size = 28
        self.img_channels = 1
        self.num_classes = 10
        self.latent_dim = 100
        self.img_shape = (self.img_size, self.img_size, self.img_channels)

    def build_discriminator(self):
        model = tf.keras.Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="sigmoid")(features)

        return tf.keras.Model(img, [validity, label])

    def build_generator(self):
        model = tf.keras.Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=(self.latent_dim + self.num_classes)))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.img_channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        #label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = concatenate([noise, label], axis=1)
        img = model(model_input)

        return tf.keras.Model([noise, label], img)

    def build_ac_gan(self, generator, discriminator):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))

        img = generator([noise, label])

        discriminator.trainable = False
        valid, target_label = discriminator(img)

        return tf.keras.Model([noise, label], [valid, target_label])
