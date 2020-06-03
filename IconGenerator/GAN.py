import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
from skimage import color
from skimage import io


def im2array(path_to_image):
    image = io.imread(os.path.abspath(path_to_image), as_gray=True)
    return np.asarray(image)  # Image.show() requires image*255


def encoder(img_1, img_2, img_3):
    pass


def transformer():
    pass


def decoder():
    pass


def discriminator():
    pass


def construct_model():
    pass

im2array("/Users/micahreich/Documents/VisualEssence/cnn_data/I_2.png")
