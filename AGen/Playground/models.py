import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import numpy as np


class ModelLib:
    def __init__(self):
        self.image_shape = (72, 72, 4)
        self.canvas_size = 72
        self.n_coords = 4

    def build_composer(self):
        latent_input = Input(shape=100)

        p = Dense(units=128)(latent_input)
        p = Dense(units=256)(p)
        p = Dense(units=512)(p)
        p = Dense(units=1024)(p)

        out = Dropout(0.4)(p)
        out = Dense(units=4)(out)  # TOP LEFT, BOTTOM RIGHT COORDINATE
        out = ReLU(max_value=self.canvas_size)(out)

        def image_paste(x):
            # Position reshaping
            positions = tf.cast(tf.reshape(x, (2, 2)), tf.int32)

            # Default tf.case lambda functions

            # Shape width and height
            w = tf.cond(
                tf.raw_ops.Greater(x=positions[1, 0], y=positions[0, 0]),
                lambda: tf.raw_ops.Cast(x=positions[1, 0] - positions[0, 0], DstT=tf.int32), lambda: 0)

            h = tf.cond(
                tf.raw_ops.Greater(x=positions[1, 1], y=positions[0, 1]),
                lambda: tf.raw_ops.Cast(x=positions[1, 1] - positions[0, 1], DstT=tf.int32), lambda: 0)

            # Shape creation
            shape = tf.zeros(shape=(h, w, 3)) + (tf.eye(3)[tf.random.uniform([], 0, 3, dtype=tf.int64)] * 250)

            # Assigning padding, checking for valid coordinates
            top_pad = tf.cond(
                tf.raw_ops.Less(x=positions[0, 1], y=positions[1, 1]),
                lambda: positions[0, 1], lambda: 0)

            bottom_pad = tf.cond(
                tf.raw_ops.Less(x=positions[0, 1], y=positions[1, 1]),
                lambda: self.canvas_size - positions[1, 1], lambda: self.canvas_size)

            left_pad = tf.cond(
                tf.raw_ops.Less(x=positions[0, 0], y=positions[1, 0]),
                lambda: positions[0, 0], lambda: 0)

            right_pad = tf.cond(
                tf.raw_ops.Less(x=positions[0, 0], y=positions[1, 0]),
                lambda: self.canvas_size - positions[1, 0], lambda: self.canvas_size)

            # Create padding tensor
            padding = [[top_pad, bottom_pad],
                       [left_pad, right_pad],
                       [0, 0]]

            s = tf.raw_ops.PadV2(input=shape, paddings=padding, constant_values=255)
            return s

        def tf_map_fn(x):
            return tf.map_fn(fn=image_paste, elems=x)

        composed_image = Lambda(tf_map_fn, name="composition_layer")(out)

        return tf.keras.Model(inputs=latent_input, outputs=composed_image)

    def build_discriminator(self):
        def conv2d(input_layer, filters, kernel_size, strides, activation='relu'):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                       padding='same', activation=activation)(input_layer)
            return BatchNormalization()(x)

        image_input = Input(shape=(72, 72, 3))

        d = conv2d(image_input, filters=64, kernel_size=3, strides=2)
        d = conv2d(d, filters=128, kernel_size=3, strides=2)
        d = conv2d(d, filters=256, kernel_size=3, strides=2)
        d = conv2d(d, filters=512, kernel_size=3, strides=2)
        d = conv2d(d, filters=512, kernel_size=3, strides=1)

        d = Flatten()(d)

        valid = Dense(units=1, activation='sigmoid')(d)

        return tf.keras.Model(inputs=image_input, outputs=valid)

    def build_full_model(self, composer, discriminator):
        latent_input = Input(shape=100, batch_size=64)

        composed_image = composer(latent_input)

        discriminator.trainable = False
        valid = discriminator(composed_image)

        return tf.keras.Model(inputs=latent_input, outputs=valid)
