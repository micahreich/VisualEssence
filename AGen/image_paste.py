import numpy as np
from PIL import Image
import os
import tensorflow as tf
import timeit
import random
import tensorflow.keras.backend as K
from matplotlib import image
from matplotlib import pyplot


def load_emojis(batch_size, path_to_data="data/"):
    images = random.sample(os.listdir(path_to_data), k=(3 * batch_size))
    images_array = []
    for i in range(len(images)):
        images_array.append(
            image.imread(path_to_data + images[i])
        )

    return np.asarray(images_array).reshape((batch_size, 3, 72, 72, 4)) * 255


canvas_size = 128
img_size = 72


def get_pixel_population(image_stack, coordinates):
    pixel_stack = []

    for i in range(image_stack.shape[0]):
        pixel_population = []
        x, y = coordinates[i]
        img = image_stack[i]  # shape (72, 72, 4), alpha = transparency

        for px in range(y, y + img_size):
            pixel_population.extend(
                [i for i in range(px * canvas_size + x, px * canvas_size + x + img_size) if img[px - y, (i - x) % canvas_size, 3] != 0]
            )

        pixel_stack.append(pixel_population)

    return pixel_stack

def image_paste(inputs, t_val=255):

    images, positions = inputs[0], inputs[1]
    batch_size = images.shape[0]

    canvas = np.ones(shape=(batch_size, canvas_size, canvas_size, 3)) * t_val

    for i in range(batch_size):
        for p in range(positions[i].shape[0]):
            _y, _x = positions[i, p].astype('uint8')
            img = images[i, p]

            for x in range(_x, _x + img_size):
                for y in range(_y, _y + img_size):
                    sx = x - _x
                    sy = y - _y
                    if img[sx, sy, 3] > 0 and not (canvas[i, x, y, 0:3] - 255).all():
                        canvas[i, x, y, 0:3] = img[sx, sy, 0:3]
    return canvas


def keras_call(input_data, t_val=255):
    if type(input_data) is not list or len(input_data) <= 1:
        raise Exception('Image composition must be called on a list of tensors. Got ' + str(input_data))

    images, positions = tf.convert_to_tensor(input_data[0], dtype=tf.float32), tf.convert_to_tensor(input_data[1])
    positions = tf.reshape(positions, [-1, 3, 2])

    batch_size = images.shape[0]

    canvas = tf.Variable(np.ones(shape=(batch_size, canvas_size, canvas_size, 3)) * t_val, dtype=tf.float32)

    for i in range(batch_size):
        for p in range(positions[i].shape[0]):
            _y, _x = tf.cast(positions[i, p], dtype=tf.int64)
            img = images[i, p]

            for x in range(_x, _x + img_size):
                for y in range(_y, _y + img_size):
                    sx = x - _x
                    sy = y - _y
                    if img[sx, sy, 3] > 0 and K.equal(K.sum(canvas[i, x, y, 0:3]), 255 * 3):
                        canvas[i, x, y, 0:3].assign(img[sx, sy, 0:3])
    return canvas


"""a = load_emojis(batch_size=2)
c = keras_call(
    input_data=[
        tf.convert_to_tensor(a),
        tf.convert_to_tensor(np.random.randint(0, 40, size=(2, 3, 2)).astype(np.float32))
    ]
)


images = c.numpy().astype(np.uint8)"""


if __name__ == '__main__':
    testLayer = tf.keras.layers.Input(shape=(2, 2), batch_size=32)
    x, y = tf.unstack(testLayer, num=2, axis=1)
    print(x.shape, y.shape)
    """composition_tensor = main()
    composition_np = composition_tensor.numpy().astype('uint8')
    Image.fromarray(composition_np).show()"""