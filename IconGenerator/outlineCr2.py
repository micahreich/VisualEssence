import cv2
import numpy as np
from PIL import Image, ImageDraw
import numpy as np

def makeOutline(path_to_image):
    img_arr = np.asarray(Image.open(path_to_image))
    img_arr = ((255 - np.asarray(img_arr))[:, :, 3])