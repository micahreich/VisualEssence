import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class I2I_AE:
    def __init__(self):
        self.img_size = 28
        self.img_channels = 1
        self.num_classes = 10
        self.latent_dim = 100
        self.img_shape = (self.img_size, self.img_size, self.img_channels)

    def build_encoder(self):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        def conv2d(layer_input, filters, kernel_size, strides, activation='relu'):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       padding="same", kernel_initializer=init)(layer_input)
            x = BatchNormalization(momentum=0.8)(x)
            x = Activation(activation)(x)
            return x

        def upconv2d(layer_input, filters, kernel_size, strides, activation='relu'):
            x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                padding="same", kernel_initializer=init)(layer_input)
            x = BatchNormalization(momentum=0.8)(x)
            x = Activation(activation)(x)
            return x

        stacked_input = Input(shape=(self.img_size, self.img_size, 2))

        encoder = conv2d(stacked_input, 64, 3, 2)
        encoder = conv2d(encoder, 128, 3, 2)
        encoder = conv2d(encoder, 256, 3, 1)
        encoder = conv2d(encoder, 512, 3, 1)
        encoder = conv2d(encoder, 1024, 3, 1)

        decoder = upconv2d(encoder, 1024, 3, 1)
        decoder = upconv2d(decoder, 512, 3, 1)
        decoder = upconv2d(decoder, 256, 3, 1)
        decoder = upconv2d(decoder, 128, 3, 2)

        def bimodal_regularizer(out):
            return 0.5 - (K.mean(K.abs(out - 0.5)))

        encoder_out = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same")(decoder)
        encoder_out = BatchNormalization(momentum=0.8)(encoder_out)
        encoder_out = Activation('sigmoid', activity_regularizer=bimodal_regularizer)(encoder_out)

        return tf.keras.Model(inputs=stacked_input, outputs=encoder_out)

    def build_decoder(self):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        def conv2d(layer_input, filters, kernel_size, strides, activation='relu'):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       padding="same", kernel_initializer=init)(layer_input)
            x = BatchNormalization(momentum=0.8)(x)
            x = Activation(activation)(x)
            return x

        def upconv2d(layer_input, filters, kernel_size, strides, activation='relu'):
            x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                padding="same", kernel_initializer=init)(layer_input)
            x = BatchNormalization(momentum=0.8)(x)
            x = Activation(activation)(x)
            return x

        composed_image = Input(shape=(self.img_size, self.img_size, 1))

        encoder = conv2d(composed_image, 64, 3, 2)
        encoder = conv2d(encoder, 128, 3, 2)
        encoder = conv2d(encoder, 256, 3, 1)
        encoder = conv2d(encoder, 512, 3, 1)
        encoder = conv2d(encoder, 1024, 3, 1)

        decoder = upconv2d(encoder, 1024, 3, 1)
        decoder = upconv2d(decoder, 512, 3, 1)
        decoder = upconv2d(decoder, 256, 3, 1)
        decoder = upconv2d(decoder, 128, 3, 2)

        def bimodal_regularizer(out):
            return 0.5 - (K.mean(K.abs(out - 0.5)))

        decoder_out = Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding="same")(decoder)
        decoder_out = BatchNormalization(momentum=0.8)(decoder_out)
        decoder_out = Activation('sigmoid', activity_regularizer=bimodal_regularizer)(decoder_out)

        return tf.keras.Model(inputs=composed_image, outputs=decoder_out)

    def build_autoencoder(self, encoder_net, decoder_net):
        stacked_input = Input(shape=(self.img_size, self.img_size, 2))

        composed_image = encoder_net(stacked_input)
        reconstructed_image = decoder_net(composed_image)

        return tf.keras.Model(inputs=stacked_input, outputs=reconstructed_image)

