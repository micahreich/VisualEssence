from skimage.morphology import convex_hull_image
from skimage import io
import os
import numpy as np
import statistics
import pickle
from numpy import asarray
from PIL import Image
from sklearn.cluster import KMeans
import shutil
from enum import Enum
import sys
from .ClusteringDataGen import DatasetGenerator


class ClusterModel:
    def __init__(self, data_directory, run_mode):
        self.run_mode = run_mode
        self.data_directory = data_directory

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

            if r_i_max > 0:
                black_rows.append(r_i_max)
            r_i_max = 0

        return statistics.median(black_rows)

    def display_data(self, labels, class_no):
        for i in range(len(labels)):
            if labels[i] == class_no:
                print(self.fnames[i][self.fnames[i].index("_")+1:self.fnames[i].index(".")] + ": " + str(labels[i]))

    def save_model(self, model):
        pickled_model = open(self.data_directory[:self.data_directory.index("VisualEssence")] + 'VisualEssence/CNN/StyleClustering/kmeans_model.pkl', 'wb')
        pickle.dump(model, pickled_model)

    def load_model(self):
        pickled_model = open(self.data_directory[:self.data_directory.index("VisualEssence")] + 'VisualEssence/CNN/StyleClustering/kmeans_model.pkl', 'rb')
        return pickle.load(pickled_model)

    def predict(self, path_to_image):
        img = Image.open(os.path.abspath(path_to_image))
        pixel_data = self.consecutive_pixels(
            ((255 - np.asarray(img))[:, :, 3])
        )

        kmeans = self.load_model()
        return kmeans.predict([[pixel_data]])[0]

    def train(self, n_samples=1200):
        ds = DatasetGenerator(n_samples, self.data_directory, "LOAD_PICKLE")
        _img_arr = ds.generate_dataset()
        print("DATASET SIZE: {}".format(len(_img_arr)))

        pixel_data = []

        for i in _img_arr:
            try:
                pixel_data.append(
                    [self.consecutive_pixels(i)]
                )
            except:
                print("ERROR IN CONSECUTIVE PX COUNT, CONTINUING...")

        kmeans = KMeans(n_clusters=3, random_state=0, max_iter=400).fit(pixel_data)
        self.save_model(kmeans)

        return kmeans


"""if __name__ == "__main__":
    cluster = ClusterModel("/nethome/mreich8/VisualEssence/data/style_data", "TRAIN")
    #cluster.train()
    print(cluster.predict("/nethome/mreich8/VisualEssence/data/style_data/I_1158669.png"))"""
