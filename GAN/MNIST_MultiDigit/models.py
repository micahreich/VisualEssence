import tensorflow as tf
from tensorflow.keras.layers import *


class cGAN:
    def __init__(self):
        self.img_size = 28
        self.img_channels = 1
        self.n_classes = 10
        self.latent_dim = 100

    def conv_block(self, _input, filters, k_size, stride):
        x = Conv2D(filters=filters, kernel_size=k_size,
                   strides=stride, padding="same",
                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))(_input)
        x = LeakyReLU(alpha=0.2)(x)
        return BatchNormalization()(x)

    def upconv_block(self, _input, filters, k_size, stride):
        x = BatchNormalization()(_input)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters, kernel_size=k_size,
                            strides=stride, padding="same",
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))(x)
        return x

    def build_discriminator(self):
        image_input = Input(shape=(self.img_size, self.img_size, self.img_channels))

        #  real/fake classification
        x = self.conv_block(image_input, 32, 5, 2)
        x = self.conv_block(x, 64, 5, 2)
        x = self.conv_block(x, 128, 5, 2)
        x = self.conv_block(x, 256, 5, 1)

        x = Flatten()(x)
        x = Dense(units=1, activation='sigmoid', name="valid_output")(x)  # predict real/fake

        #  one-hot label prediction
        z = self.conv_block(image_input, 32, 5, 2)
        z = self.conv_block(z, 64, 5, 2)
        z = self.conv_block(z, 128, 5, 2)
        z = self.conv_block(z, 256, 5, 1)
        z = self.conv_block(z, 512, 5, 1)

        z = Flatten()(z)
        z = Dense(units=self.n_classes, activation='sigmoid', name="digit_output")(z)  # predict one-hot labels

        return tf.keras.Model(inputs=image_input, outputs=[x, z])

    def build_generator(self):
        img_resize = self.img_size // 4
        latent_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(self.n_classes,))

        x = concatenate([latent_input, label_input], axis=1)
        x = Dense(img_resize*img_resize*128)(x)
        x = Reshape(target_shape=(img_resize, img_resize, 128))(x)

        x = self.upconv_block(x, 128, 5, 2)
        x = self.upconv_block(x, 64, 5, 2)
        x = self.upconv_block(x, 32, 5, 1)
        x = self.upconv_block(x, 1, 5, 1)
   
        x = Activation('sigmoid')(x)  # generated image

        return tf.keras.Model([latent_input, label_input], x)

    def build_cgan(self, generator, discriminator):
        latent_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(self.n_classes,))

        img = generator([latent_input, label_input])

        discriminator.trainable = False
        valid, class_pred = discriminator(img)

        return tf.keras.Model(inputs=[latent_input, label_input], outputs=[valid, class_pred])
