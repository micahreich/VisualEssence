import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import numpy as np


class ImagePaste(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(ImagePaste, self).__init__(**kwargs)
        self.img_size = 72
        self.canvas_size = 128

    def call(self, input_data, t_val=255):
        if type(input_data) is not list or len(input_data) <= 1:
            raise Exception('Image composition must be called on a list of tensors. Got ' + str(input_data))

        images, positions = tf.convert_to_tensor(input_data[0], dtype=tf.float32), tf.convert_to_tensor(input_data[1])
        positions = tf.reshape(positions, [-1, 3, 2])

        batch_size = images.shape[0]

        canvas = tf.Variable(np.ones(shape=(batch_size, self.canvas_size, self.canvas_size, 3)) * t_val,
                             dtype=tf.float32,
                             trainable=False)

        for i in range(batch_size):
            for p in range(positions[i].shape[0]):
                _y, _x = tf.cast(positions[i, p], dtype=tf.int64)
                img = images[i, p]

                for x in range(_x, _x + self.img_size):
                    for y in range(_y, _y + self.img_size):
                        sx = x - _x
                        sy = y - _y
                        if img[sx, sy, 3] > 0 and K.equal(K.sum(canvas[i, x, y, 0:3]), 255 * 3):
                            canvas[i, x, y, 0:3].assign(img[sx, sy, 0:3])
        return canvas


class ImageVisibility(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ImageVisibility, self).__init__(**kwargs)
        self.img_size = 72
        self.canvas_size = 128


class ModelLib:
    def __init__(self):
        self.image_shape = (72, 72, 4)
        self.img_size = 72
        self.composition_shape = (128, 128, 3)
        self.n_coords = 6

    def build_position_pred(self):
        def conv2d(input_layer, filters, kernel_size, strides, activation='relu'):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                       padding='same', activation=activation)(input_layer)
            return BatchNormalization()(x)

        stacked_input = Input(shape=(3, 72, 72, 4))
        image_1, image_2, image_3 = tf.unstack(stacked_input, num=3, axis=1)

        x = conv2d(image_1, filters=32, kernel_size=7, strides=2)
        y = conv2d(image_2, filters=32, kernel_size=7, strides=2)
        z = conv2d(image_3, filters=32, kernel_size=7, strides=2)

        img_concat = Concatenate()([x, y, z])

        p = conv2d(img_concat, filters=64, kernel_size=3, strides=2)
        p = conv2d(p, filters=128, kernel_size=3, strides=2)
        p = conv2d(p, filters=256, kernel_size=3, strides=2)
        p = conv2d(p, filters=512, kernel_size=3, strides=1)
        p = conv2d(p, filters=1024, kernel_size=3, strides=1)

        p = Flatten()(p)

        p = Dense(units=2048, activation='relu')(p)
        p = Dropout(0.4)(p)

        coords = Dense(units=self.n_coords, activation='relu')(p)

        return tf.keras.Model(inputs=stacked_input, outputs=coords)

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

    def build_composition_pipeline(self, position_model):
        def visiblity_regularizer(x):
            return tf.math.where(x)

        stacked_input = Input(shape=(3, 72, 72, 4))

        position_prediction = position_model(stacked_input)
        composed_image = ImagePaste(trainable=False, activity_regularizer=visiblity_regularizer)(
            input_data=[stacked_input, position_prediction]
        )

        resized_composition = tf.image.resize(composed_image, [self.img_size, self.img_size])

        return tf.keras.Model(inputs=stacked_input, outputs=resized_composition)

    def build_full_model(self, composer, discriminator):
        stacked_input = Input(shape=(3, 72, 72, 4))

        composed_image = composer(stacked_input)
        valid = discriminator(composed_image)

        return tf.keras.Model(inputs=stacked_input, outputs=valid)

