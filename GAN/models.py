import os
import json
import sys
import requests
from io import BytesIO
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


class Text2Image:
    def __init__(self):
        self.noise_input_dim = 100
        self.text_input_dim = 300  # GloVe word embeddings are 300d
        self.projected_embed_dim = 128
        self.image_size = 200

    def upconv_block(self, _input, filters, kernel_size, strides, activation=None, bn=True):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        g = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding="same",
                            activation=activation,
                            kernel_initializer=init)(_input)

        if bn:
            g = BatchNormalization()(g)
            g = LeakyReLU(alpha=0.2)(g)

        return g

    def conv_block(self, _input, filters, kernel_size, strides, activation=None, bn=True):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        d = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding="same",
                   activation=activation,
                   kernel_initializer=init)(_input)

        if bn:
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)

        return d

    def build_generator(self):
        noise_input = Input(shape=(self.noise_input_dim,), name="Z_Input")
        text_input = Input(shape=(self.text_input_dim,), name="Text_Input")

        init_1 = tf.keras.initializers.RandomNormal(stddev=0.02)
        init_2 = tf.keras.initializers.RandomNormal(stddev=0.02)

        text_embedding = Dense(units=128, name="Text_Embed", kernel_initializer=init_1)(text_input)
        text_embedding = LeakyReLU(alpha=0.2)(text_embedding)

        z_concat = concatenate([noise_input, text_embedding], name="Text_Noise_Concat")

        g = Dense(units=5*5*64*16, activation='relu', kernel_initializer=init_2)(z_concat)
        g = Reshape(target_shape=(5, 5, 64*16))(g)
        g = BatchNormalization()(g)
        g = ReLU()(g)

        g = self.upconv_block(g, 64 * 8, (10, 10), (2, 2))
        g = self.upconv_block(g, 64 * 4, (20, 20), (2, 2))
        g = self.upconv_block(g, 64 * 2, (40, 40), (2, 2))
        g = self.upconv_block(g, 1, (80, 80), (5, 5), activation='tanh', bn=False)

        model = tf.keras.Model([text_input, noise_input], g)

        return model

    def build_discriminator(self):
        image_input = Input(shape=(200, 200, 1))
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        d = self.conv_block(image_input, 64, (2, 2), (2, 2), bn=False)
        d = LeakyReLU()(d)

        d = self.conv_block(d, 64 * 2, (2, 2), (2, 2))
        d = self.conv_block(d, 64 * 4, (2, 2), (3, 3))
        d = self.conv_block(d, 64 * 8, (2, 2), (3, 3))

        text_input = Input(shape=(self.text_input_dim,), name="Text_Input")

        text_embedding = Dense(units=128, name="Text_Embed", kernel_initializer=init)(text_input)
        text_embedding = LeakyReLU(alpha=0.2)(text_embedding)

        expanded_embedding = tf.keras.backend.expand_dims(text_embedding, 1)
        expanded_embedding = tf.keras.backend.expand_dims(expanded_embedding, 2)

        tiled_embeddings = tf.keras.backend.tile(expanded_embedding, [1, 6, 6, 1])

        concat = concatenate([d, tiled_embeddings], axis=3, name="Spatial_Concat")

        d = self.conv_block(concat, 64 * 4, (1, 1), (1, 1))
        d = self.conv_block(d, 64 * 4, (6, 6), (2, 2))

        d = Flatten()(d)
        d = Dense(units=1, activation="sigmoid")(d)

        model = tf.keras.Model([image_input, text_input], d)

        return model

    def build_combined(self, _generator, _discriminator):
        _discriminator.trainable = False

        noise_input = Input(shape=(self.noise_input_dim,))
        text_input = Input(shape=(self.text_input_dim,))

        gen_img = _generator([text_input, noise_input])
        valid = _discriminator([gen_img, text_input])

        model = tf.keras.Model([text_input, noise_input], valid)

        return model
