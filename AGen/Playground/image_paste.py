import numpy as np
from PIL import Image
import tensorflow as tf

canvas_size = 72


def square_paste(positions):
    positions = tf.reshape(tf.convert_to_tensor(positions), (2, 2))
    w, h = positions[1, 0] - positions[0, 0], positions[1, 1] - positions[0, 1]

    shape = tf.zeros(shape=(h, w, 3)) + (tf.eye(3)[tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int64)] * 250)

    padding = [[positions[0, 1], canvas_size - positions[1, 1]],
               [positions[0, 0], tf.constant(canvas_size) - positions[1, 0]],
               [0, 0]]

    return tf.pad(shape, padding, mode="CONSTANT", constant_values=255)


c = square_paste([(5, 8), (20, 60)])
Image.fromarray(np.asarray(c).astype("uint8")).show()
