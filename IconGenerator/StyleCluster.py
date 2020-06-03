from skimage.morphology import convex_hull_image
from skimage import io
import os
import numpy as np
import statistics
import pickle
from numpy import asarray
from PIL import Image
from sklearn.cluster import KMeans
from enum import Enum, auto
import sys
import IconDataGen


class RunMode(Enum):
    DOWNLOAD_PICKLE_CLUSTER = auto()
    PICKLE_CLUSTER = auto()
    CLUSTER = auto()


class StyleCluster():
    def __init__(self, path_to_images, run_mode):
        self.path_to_images = path_to_images
        self.run_mode = run_mode
        self.fnames = []

    def consecutive_pixels(self, array):
        img = array

        r_i_max = 0
        black_rows = []
        for i in range(len(img)):
            r_i_current = 0
            for j in range(len(img[0])):
                if img[i][j] != 255:
                    r_i_current += 1
                    if r_i_current > r_i_max:
                        r_i_max = r_i_current
                else:
                    r_i_current = 0
                    if len(img[0]) - j < r_i_max:
                        break

            black_rows.append(r_i_max)
            r_i_max = 0
        return statistics.median(black_rows)

    def pickle_dataset(self, img_arr):
        pickled_images = open(os.path.abspath(self.path_to_images + '/pkl_images.pkl'), 'wb')
        pickle.dump(img_arr, pickled_images)

    def get_images(self):
        _img_arr = []

        counter = 1
        for i in self.fnames:
            if counter % 20 == 0:
                print("GRABBED " + str(counter) + " IMAGES")
            try:
                img = io.imread(os.path.abspath(self.path_to_images + "/" + i), as_gray=True)
                _img_arr.append(asarray(img * 255))
            except ValueError:
                print("VALUE ERROR, CONTINUING...")
                self.fnames.remove(i)

            counter += 1

        self.pickle_dataset(_img_arr)

    def display_data(self, labels, class_no):
        for i in range(len(labels)):
            if labels[i] == class_no:
                print(self.fnames[i].replace(".png", "") + ":" + str(labels[i]))

    def cluster_images(self, n_samples=10):
        if self.run_mode == RunMode.DOWNLOAD_PICKLE_CLUSTER:
            DataGen = IconDataGen.DatasetGenerator(n_samples, self.path_to_images)
            DataGen.get_images(self.path_to_images)
            self.fnames = os.listdir(os.path.abspath(self.path_to_images))
            self.get_images()
            self.fnames = os.listdir(os.path.abspath(self.path_to_images))
        elif self.run_mode == RunMode.PICKLE_CLUSTER:
            self.fnames = os.listdir(os.path.abspath(self.path_to_images))
            self.get_images()
        else:
            self.fnames = os.listdir(os.path.abspath(self.path_to_images))

        pickled_images = open(os.path.abspath(self.path_to_images + '/pkl_images.pkl'), 'rb')
        self.fnames.remove("pkl_images.pkl")

        _img_arr = pickle.load(pickled_images)
        pixel_data = []

        for i in _img_arr:
            pixel_data.append(
                [self.consecutive_pixels(i)]
            )

        kmeans = KMeans(n_clusters=3, random_state=0).fit(pixel_data)

        return kmeans


if __name__ == "__main__":
    cluster = StyleCluster("/Users/micahreich/Documents/VisualEssence/style_data", RunMode.CLUSTER)
    model = cluster.cluster_images()
    cluster.display_data(model.labels_, 0)
