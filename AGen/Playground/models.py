import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import numpy as np


class ImagePaste(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(ImagePaste, self).__init__(**kwargs)
        self.canvas_size = 72

    def build(self, input_shape):
        self.batch_s = input_shape[0]

        super(ImagePaste, self).build(input_shape)
    
    def call(self, input_data, t_val=255):
        positions = tf.convert_to_tensor(input_data)
        positions = tf.reshape(positions, [-1, 2, 2])

        canvas = tf.Variable(tf.ones([self.batch_s, 72, 72, 3]) * t_val,
                             dtype=tf.float32,
                             trainable=False, validate_shape=True)

        for i in tf.range(0, self.batch_s):
            color = tf.convert_to_tensor(np.eye(3)[np.random.choice(3)] * 250, dtype=tf.float32)
            tl, br = tf.cast(positions[i], dtype=tf.int64)

            for r in tf.range(min(tl[0], self.canvas_size), min(br[0], self.canvas_size)):
                for c in tf.range(min(tl[1], self.canvas_size), min(br[1], self.canvas_size)):
                    canvas[i, r, c, :].assign(color)

        return tf.convert_to_tensor(canvas)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.canvas_size, self.canvas_size, 3)


class ModelLib:
    def __init__(self):
        self.image_shape = (72, 72, 4)
        self.img_size = 72
        self.n_coords = 6

    def build_composer(self):
        latent_input = Input(shape=100, batch_size=64)

        p = Dense(units=128)(latent_input)
        p = Dense(units=256)(p)
        p = Dense(units=512)(p)
        p = Dense(units=1024)(p)

        out = Dropout(0.4)(p)
        out = Dense(units=4, activation='relu')(out)  # TOP LEFT, BOTTOM RIGHT COORDINATE

        composed_image = ImagePaste(trainable=False, dynamic=True)(out)

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

