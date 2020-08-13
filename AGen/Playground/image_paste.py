import numpy as np
from PIL import Image
import tensorflow as tf

canvas_size = 72


def image_paste(x):
    positions = tf.cast(tf.reshape(x, (2, 2)), tf.int32)

    zero_default = lambda: 0
    canvas_default = lambda: canvas_size

    width_fn = lambda: tf.cast(positions[1, 0] - positions[0, 0], tf.int32)
    w = tf.case(
        [(tf.greater(positions[1, 0], positions[0, 0]), width_fn)],
        default=zero_default)

    height_fn = lambda: tf.cast(positions[1, 1] - positions[0, 1], tf.int32)
    h = tf.case(
        [(tf.greater(positions[1, 1], positions[0, 1]), height_fn)],
        default=zero_default)

    shape = tf.zeros(shape=(h, w, 3)) + (tf.eye(3)[tf.random.uniform([], 0, 3, dtype=tf.int64)] * 250)

    top_pad_fn = lambda: positions[0, 1]
    top_pad = tf.case(
        [(tf.less(positions[0, 1], positions[1, 1]), top_pad_fn)],
        default=zero_default)

    bottom_pad_fn = lambda: canvas_size - positions[1, 1]
    bottom_pad = tf.case(
        [(tf.less(positions[0, 1], positions[1, 1]), bottom_pad_fn)],
        default=canvas_default)

    left_pad_fn = lambda: positions[0, 0]
    left_pad = tf.case(
        [(tf.less(positions[0, 0], positions[1, 0]), left_pad_fn)],
        default=zero_default)

    right_pad_fn = lambda: canvas_size - positions[1, 0]
    right_pad = tf.case(
        [(tf.less(positions[0, 0], positions[1, 0]), right_pad_fn)],
        default=canvas_default)

    padding = [[top_pad, bottom_pad],
               [left_pad, right_pad],
               [0, 0]]

    s = tf.pad(shape, padding, mode="CONSTANT", constant_values=255)
    print(s.shape)
    return s


c = image_paste(tf.constant([2, 0, 23, 7]))
Image.fromarray(np.asarray(c).astype("uint8")).show()
