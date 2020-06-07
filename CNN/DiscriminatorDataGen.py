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
from IconGeneration import ConvexHull
from StyleClustering import StyleCluster


class DatasetGenerator:
    def __init__(self, n_samples, data_directory, run_mode):
        try:
            os.mkdir(data_directory)
        except FileExistsError:
            print(f'WARNING: DIRECTORY {data_directory} ALREADY EXISTS')

        self.n_samples = n_samples
        self.data_directory = data_directory
        self.run_mode = run_mode
        self.style_class = 0

    def download_images(self):
        print("\nSTARTING ICON COLLECTION")

        current_count = 0
        icon_ids = []

        style_classifier = StyleCluster.ClusterModel(self.data_directory[:self.data_directory.index("data")] + "/data/style_data", "PREDICT")

        while current_count < self.n_samples:
            random_id = randint(1, 3368879)
            while random_id in icon_ids:
                random_id = randint(1, 3368879)

            img_url = "https://static.thenounproject.com/png/{}-200.png".format(random_id)

            with open(os.path.abspath(self.data_directory + "/I_" + str(random_id) + ".png"), 'wb') as f:
                f.write(requests.get(img_url).content)
            f.close()

            if style_classifier.predict(self.data_directory + "/I_" + str(random_id) + ".png") != self.style_class:
                os.remove(self.data_directory + "/I_" + str(random_id) + ".png")
            else:
                current_count += 1
                icon_ids.append(random_id)

                if current_count % 1000 == 0:
                    print(f"DOWNLOADED {current_count} IMAGES, {(current_count / self.n_samples) * 100}% COMPLETE")

    def generate_pos_vec(self, icon_size=120):
        position_vector = []

        vec_1 = sample(range(int(icon_size / 2), int(200 - (icon_size / 2))), 2)  # (x, y)
        position_vector.append(vec_1)

        for i in range(2):
            radius = randint(int(icon_size / 2), int(icon_size) + 20)
            angle = random() * 6.28319  # 0 to 2pi radians

            while (position_vector[i][0] + radius * math.cos(angle) + icon_size / 2 > 200 or position_vector[i][0] + radius * math.cos(angle) - icon_size / 2 < 0) or \
                  (position_vector[i][1] + radius * math.sin(angle) + icon_size / 2 > 200 or position_vector[i][1] + radius * math.sin(angle) - icon_size / 2 < 0):
                radius = randint(int(icon_size / 2), int(icon_size) + 20)
                angle = random() * 6.28319

            vec = [int(position_vector[i][0] + radius * math.cos(angle)),
                   int(position_vector[i][1] + radius * math.sin(angle))]
            position_vector.append(vec)

        return position_vector

    def create_negative_sample(self, icons, ID):
        try:
            ConvexHull.convex_hull(self.data_directory,
                                   icons,
                                   self.generate_pos_vec(), ID)
        except PIL.UnidentifiedImageError as e:
            print("IMAGE ERROR, COULD NOT FIND: ", e)
        except ValueError:
            print("IMAGE ERROR, CORRUPTED IMG")

    def create_samples(self):
        print("\n STARTING NEGATIVE SAMPLE CREATION")

        files = os.listdir(os.path.abspath(self.data_directory))
        while ".DS_Store" in files:
            files.remove(".DS_Store")

        icon_id = 1

        for i in range(1, int(self.n_samples / 4) + 1):
            if i % 1000 == 0:
                print(f"GENERATED {i} NEGATIVE SAMPLES, {(i / (self.n_samples / 4)) * 100}% COMPLETE")

            images = sample(population=files, k=3)
            files.remove(images[0])
            files.remove(images[1])
            files.remove(images[2])

            self.create_negative_sample(images, icon_id)
            icon_id += 1

            os.remove(self.data_directory + "/" + images[0])
            os.remove(self.data_directory + "/" + images[1])
            os.remove(self.data_directory + "/" + images[2])

    def pickle_dataset(self):
        print("\n PICKLING DATASET")

        files = os.listdir(self.data_directory)
        i_cnt = 0
        r_cnt = 0
        for i in files:
            if i[0].upper() == "I":
                i_cnt += 1
            else:
                r_cnt += 1
        print(f"POS IMG COUNT {i_cnt}")
        print(f"NEG IMG COUNT {r_cnt}")

        pickled_images = open((self.data_directory + '/pkl_images.pkl'), 'wb')
        pickled_labels = open((self.data_directory + '/pkl_labels.pkl'), 'wb')

        images = []
        labels = []

        for i in os.listdir(self.data_directory):
            try:
                if i[0].upper() == "I":
                    current_image = self.data_directory + f"/{i}"
                    _img_1 = Image.open(current_image)
                    _img_arr = ((255 - np.asarray(_img_1))[:, :, 3])

                    images.append(_img_arr)
                    labels.append(1)
                    print("POSITIVE IMAGE, UNIQUE: ", len(np.unique(_img_arr)))
                elif i[0].upper() == "R":
                    current_image = self.data_directory + f"/{i}"
                    _img_1 = Image.open(current_image)
                    _img_arr = ((np.asarray(_img_1))[:, :, 2])

                    images.append(_img_arr)
                    labels.append(0)
                    print("NEGATIVE IMAGE, UNIQUE: ", len(np.unique(_img_arr)))

            except PIL.UnidentifiedImageError:
                print("COULD NOT FIND IMAGE, CONTINUING...")

        pickle.dump(images, pickled_images)
        pickle.dump(labels, pickled_labels)

        print(f"TOTAL IMAGE COUNT: {len(images)}")
        print(f"IMAGE ARRAY SHAPE: {asarray(images).shape}")

    def load_pickled_dataset(self):
        print("\n LOADING PICKLED DATASET")

        pickled_images = open((self.data_directory + '/pkl_images.pkl'), 'rb')
        pickled_labels = open((self.data_directory + '/pkl_labels.pkl'), 'rb')

        images = pickle.load(pickled_images)
        labels = pickle.load(pickled_labels)

        _images = asarray(images)
        _labels = asarray(labels)

        return _images, _labels

    def generate_dataset(self):
        if self.run_mode == "DOWNLOAD":
            self.download_images()
            print("\nICON COLLECTION COMPLETED SUCCESSFULLY")

        elif self.run_mode == "NEG_SAMPLE":
            self.create_samples()
            print("NEGATIVE SAMPLE CREATION COMPLETED SUCCESSFULLY")

        elif self.run_mode == "PICKLE":
            self.pickle_dataset()
            print("CREATED PICKLED DATASET SUCCESSFULLY")

        elif self.run_mode == "LOAD_PICKLE":
            data = self.load_pickled_dataset()
            print("GENERATING DATA SPLIT")

            split_point = int(0.8 * (self.n_samples / 2))

            images, labels = sklearn.utils.shuffle(data[0], data[1])

            images_train = images[0:split_point]
            labels_train = labels[0:split_point]

            images_test = images[split_point:]
            labels_test = labels[split_point:]

            print("Train images, labels: ", len(images_train),
                  "\nImage array shape: ", images_train.shape,
                  "\nLabel array shape: ", labels_train.shape)

            print("Test images, labels: ", len(images_test),
                  "\nImage array shape: ", images_test.shape,
                  "\nLabel array shape: ", labels_test.shape)

            return images_train, labels_train, images_test, labels_test


if __name__ == "__main__":
    # /nethome/mreich8/VisualEssence/data/CNN/cnn_data
    # RunMode options: DOWNLOAD, NEG_SAMPLE, PICKLE, LOAD_PICKLE
    DS = DatasetGenerator(8, "/Users/micahreich/Documents/VisualEssence/data/cnn_data", "LOAD_PICKLE")
    DS.generate_dataset()
