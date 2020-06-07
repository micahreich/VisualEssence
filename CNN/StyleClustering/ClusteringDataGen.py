import requests
from PIL import Image
import PIL
from random import randint, sample, random
import os
import numpy as np
from numpy import asarray
import pickle
import math
import shutil
import sklearn
from enum import Enum


class DatasetGenerator:
    def __init__(self, n_samples, data_directory, run_mode):
        try:
            os.mkdir(data_directory)
        except FileExistsError:
            print(f'WARNING: DIRECTORY {data_directory} ALREADY EXISTS')

        self.n_samples = n_samples
        self.data_directory = data_directory
        self.run_mode = run_mode

    def download_images(self):
        print("\nSTARTING ICON COLLECTION")

        icon_ids = sample(range(1, 3368879), self.n_samples)  # 3368879 icons in NP database

        for i in range(len(icon_ids)):
            if i+1 % 1000 == 0:
                print(f"DOWNLOADED {i+1} IMAGES, {(i+1 / self.n_samples) * 100}% COMPLETE")

            img_url = "https://static.thenounproject.com/png/{}-200.png".format(icon_ids[i])

            with open(os.path.abspath(self.data_directory + "/I_" + str(icon_ids[i]) + ".png"), 'wb') as f:
                f.write(requests.get(img_url).content)
            f.close()

    def pickle_dataset(self):
        pickled_images = open(os.path.abspath(self.data_directory + '/pkl_images.pkl'), 'wb')

        images = []
        for i in os.listdir(self.data_directory):
            try:
                current_image = self.data_directory + f"/{i}"
                _img_1 = Image.open(current_image)
                _img_arr = ((255 - np.asarray(_img_1))[:, :, 3])

                images.append(_img_arr)

            except PIL.UnidentifiedImageError:
                print("COULD NOT FIND IMAGE, CONTINUING...")

        pickle.dump(images, pickled_images)

    def load_dataset(self):
        pickled_images = open((self.data_directory + '/pkl_images.pkl'), 'rb')
        images = pickle.load(pickled_images)

        return asarray(images)

    def generate_dataset(self):
        if self.run_mode == "DOWNLOAD":
            self.download_images()
            print("\nICON COLLECTION COMPLETED SUCCESSFULLY")

        elif self.run_mode == "PICKLE":
            self.pickle_dataset()
            print("CREATED PICKLED DATASET SUCCESSFULLY")

        elif self.run_mode == "LOAD_PICKLE":
            data = self.load_dataset()
            print("LOADED PICKLED DATASET SUCCESSFULLY")

            return data


"""if __name__ == "__main__":
    DS = DatasetGenerator(40, "/Users/micahreich/Documents/VisualEssence/data/style_data", "PICKLE")
    DS.generate_dataset()"""