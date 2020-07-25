import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa


class I2I_GAN:
    def __init__(self):
        self.img_size = 28
        self.img_channels = 1
        self.num_classes = 10
        self.latent_dim = 100
        self.img_shape = (self.img_size, self.img_size, self.img_channels)

    def build_generator(self):
        def conv2d(layer_input, filters, kernel_size, strides, activation='relu'):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(layer_input)
            x = tfa.layers.InstanceNormalization(axis=1)(x)
            x = Activation(activation)(x)
            return x

        def resnet(layer_input, filters, kernel_size, strides):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(layer_input)
            x = tfa.layers.InstanceNormalization(axis=1)(x)
            x = Activation('relu')(x)

            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
            x = tfa.layers.InstanceNormalization(axis=1)(x)
            x = Add()([x, layer_input])
            return x

        def upconv2d(layer_input, filters, kernel_size, strides):
            x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(layer_input)
            x = tfa.layers.InstanceNormalization(axis=1)(x)
            x = Activation('relu')(x)
            return x

        # Takes in image x, image y and produces composed image c
        image_input_1 = Input(shape=self.img_shape)
        image_input_2 = Input(shape=self.img_shape)

        # 3rd axis concat of input images
        img_concat = concatenate([image_input_1, image_input_2], axis=3)

        # Convolution of image concat into latent space
        encoder = conv2d(img_concat, filters=64, kernel_size=7, strides=1)
        encoder = conv2d(encoder, filters=128, kernel_size=3, strides=2)
        encoder = conv2d(encoder, filters=256, kernel_size=3, strides=2)

        # Image transformation
        encoder = resnet(encoder, filters=256, kernel_size=3, strides=1)
        encoder = resnet(encoder, filters=256, kernel_size=3, strides=1)
        encoder = resnet(encoder, filters=256, kernel_size=3, strides=1)
        encoder = resnet(encoder, filters=256, kernel_size=3, strides=1)

        # Up-convolution of latent space into composed image
        decoder = upconv2d(encoder, filters=128, kernel_size=3, strides=2)
        decoder = upconv2d(decoder, filters=64, kernel_size=3, strides=2)
        out_image = conv2d(decoder, filters=1, kernel_size=7, strides=1, activation='tanh')

        return tf.keras.Model(inputs=[image_input_1, image_input_2], outputs=out_image)

    def build_discriminator(self):
        def conv2d(layer_input, filters, kernel_size, strides):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(layer_input)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(rate=0.3)(x)
            return x

        image_input = Input(shape=self.img_shape)

        disc = conv2d(image_input, filters=64, kernel_size=3, strides=2)
        disc = conv2d(disc, filters=128, kernel_size=3, strides=2)
        disc = conv2d(disc, filters=256, kernel_size=3, strides=2)
        disc = conv2d(disc, filters=512, kernel_size=3, strides=2)
        flatten = Flatten()(disc)

        validity = Dense(1, activation="sigmoid", name="valid")(flatten)
        label = Dense(self.num_classes, activation="sigmoid", name="label")(flatten)

        return tf.keras.Model(inputs=image_input, outputs=[validity, label])

    def build_gan(self, generator, discriminator):
        image_input_1 = Input(shape=self.img_shape)
        image_input_2 = Input(shape=self.img_shape)

        img = generator([image_input_1, image_input_2])

        discriminator.trainable = False
        valid, target_label = discriminator(img)

        return tf.keras.Model(inputs=[image_input_1, image_input_2], outputs=[valid, target_label])
