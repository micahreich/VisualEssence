import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class I2I_AE:
    def __init__(self):
        self.img_size = 32
        self.img_channels = 1
        self.num_classes = 10
        self.latent_dim = 100
        self.img_shape = (self.img_size, self.img_size, self.img_channels)
        self.gf = 64

    def build_encoder(self):
        def conv2d(layer_input, filters, f_size=4, strides=1, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, strides=1, dropout_rate=0):
            """Layers used during upsampling"""
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=strides, padding='same', activation='relu')(layer_input)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input(shape=(self.img_size, self.img_size, 2))

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False, strides=2)
        d2 = conv2d(d1, self.gf*2, strides=2)
        d3 = conv2d(d2, self.gf*4, strides=2)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)

        l = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(l, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2, strides=2)
        u6 = deconv2d(u5, d1, self.gf, strides=2)

        out_image = Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='relu')(u6)

        return tf.keras.Model(d0, out_image)

    def build_decoder(self):
        def conv2d(layer_input, filters, f_size=4, strides=1, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, strides=1, dropout_rate=0):
            """Layers used during upsampling"""
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=strides, padding='same', activation='relu')(layer_input)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input(shape=(self.img_size, self.img_size, 1))

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False, strides=2)
        d2 = conv2d(d1, self.gf*2, strides=2)
        d3 = conv2d(d2, self.gf*4, strides=2)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)

        l = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(l, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2, strides=2)
        u6 = deconv2d(u5, d1, self.gf, strides=2)

        out_image = Conv2DTranspose(2, kernel_size=4, strides=2, padding='same', activation='relu')(u6)

        return tf.keras.Model(d0, out_image)

    def build_autoencoder(self, encoder_net, decoder_net):
        stacked_input = Input(shape=(self.img_size, self.img_size, 2))

        composed_image = encoder_net(stacked_input)
        reconstructed_image = decoder_net(composed_image)

        return tf.keras.Model(inputs=stacked_input, outputs=reconstructed_image)


I2I_AE().build_encoder()