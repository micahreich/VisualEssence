from skimage.morphology import convex_hull_image
from skimage import io
from skimage.util import invert
from PIL import Image
import os
import numpy as np
import statistics
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage import measure
from numpy import asarray
from sklearn.cluster import KMeans


def consecutive_pixels(array):
    img = array

    r_i_max = 0
    black_rows = []
    for i in range(len(img)):
        r_i_current = 0
        for j in range(len(img[0])):
            if img[i][j] != 1:
                r_i_current += 1
                if r_i_current > r_i_max:
                    r_i_max = r_i_current
            else:
                r_i_current = 0
                if len(img[0]) - j < r_i_max:
                    break

        black_rows.append(r_i_max)
        r_i_max = 0

    print("ROW MEDIAN ", statistics.median(black_rows))
    return statistics.median(black_rows)


def cluster_images(path_to_images, n_samples=100):
    _img_arr = []
    pixel_data = []

    for i in os.listdir(os.path.abspath(path_to_images)):
        img = io.imread(os.path.abspath(path_to_images + "/" + i), as_gray=True)
        _img_arr.append(asarray(img))

    print(os.listdir(os.path.abspath(path_to_images)))

    for i in _img_arr:
        pixel_data.append(
            [consecutive_pixels(i)]
        )

    kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_data)
    print(kmeans.labels_)


cluster_images('test_icons')