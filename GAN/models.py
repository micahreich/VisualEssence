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
        g = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding="same",
                            activation=activation)(_input)

        if bn:
            g = BatchNormalization()(g)
            g = ReLU()(g)

        return g

    def conv_block(self, _input, filters, kernel_size, strides, activation=None, bn=True):
        d = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding="same",
                   activation=activation)(_input)

        if bn:
            d = BatchNormalization()(d)
            d = LeakyReLU()(d)

        return d

    def build_generator(self):
        noise_input = Input(shape=(self.noise_input_dim,), name="Z_Input")
        text_input = Input(shape=(self.text_input_dim,), name="Text_Input")

        text_embedding = Dense(units=128, name="Text_Embed")(text_input)
        text_embedding = LeakyReLU()(text_embedding)

        z_concat = concatenate([noise_input, text_embedding], name="Text_Noise_Concat")

        g = Dense(units=5*5*64*16, activation='relu')(z_concat)
        g = Reshape(target_shape=(5, 5, 50*16))(g)
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

        d = self.conv_block(image_input, 64, (2, 2), (2, 2), bn=False)
        d = LeakyReLU()(d)

        d = self.conv_block(d, 64 * 2, (2, 2), (2, 2))
        d = self.conv_block(d, 64 * 4, (2, 2), (3, 3))
        d = self.conv_block(d, 64 * 8, (2, 2), (3, 3))

        text_input = Input(shape=(self.text_input_dim,), name="Text_Input")

        text_embedding = Dense(units=128, name="Text_Embed")(text_input)
        text_embedding = LeakyReLU()(text_embedding)

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

    def build_combined(self, generator, discriminator):
        noise_input = Input(shape=(self.noise_input_dim,))
        text_input = Input(shape=(self.text_input_dim,))

        gen_img = generator([text_input, noise_input])

        discriminator.trainable = False
        valid = discriminator([gen_img, text_input])

        model = tf.keras.Model([text_input, noise_input], valid)

        return model
