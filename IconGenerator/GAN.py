import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
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

def img_manip(p2img):
    img = Image.open(p2img)
    img2 = (255-np.asarray(img))[:, :, 3]
    print(img2[0][0])
    Image.fromarray(img2).show()

img_manip("/Users/micahreich/Documents/VisualEssence/style_data/I_726077.png")
