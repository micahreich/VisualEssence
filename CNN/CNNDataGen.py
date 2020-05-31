import requests
from PIL import Image
from random import randint
from random import sample
from random import choice
import os
import numpy as np
from numpy import asarray
import sklearn
import tensorflow as tf
import pickle
import math
import shutil


class DatasetGenerator:
    def __init__(self, n_samples):
        try:
            os.mkdir("cnn_data")
        except FileExistsError:
            print("WARNING: directory cnn_data already exists")

        self.n_samples = n_samples

    def get_images(self, save_directory):

        img_ids = []
        icon_id = 0

        for i in range(self.n_samples):
            rand_icon_id = randint(0, 3368879)

            while rand_icon_id in img_ids:
                rand_icon_id = randint(0, 3368879)

            img_ids.append(rand_icon_id)

            img_url = "https://static.thenounproject.com/png/{}-200.png".format(rand_icon_id)

            with open(save_directory + "/I_" + str(icon_id) + ".png", 'wb') as f:
                f.write(requests.get(img_url).content)
            f.close()

            icon_id += 1

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def create_negative_sample(self, path_to_img_1, path_to_img_2, path_to_img_3, icon_id):

        # Assuming 200 x 200 images rescaled to 140 x 140 to fit in different locations
        # within the new 200 x 200 random amalgamation

        positions = []

        background = Image.open("blank_icon.png")
        _img_1 = Image.open(path_to_img_1).resize((100, 100), Image.ANTIALIAS)
        _img_2 = Image.open(path_to_img_2).resize((100, 100), Image.ANTIALIAS)
        _img_3 = Image.open(path_to_img_3).resize((100, 100), Image.ANTIALIAS)

        # Assuming 200 x 200 background
        rand_placement_vector_1 = sample(range(50, 150), 2)
        background.paste(_img_1, (rand_placement_vector_1[0], rand_placement_vector_1[1]), _img_1)
        positions.append([rand_placement_vector_1[0]-50, rand_placement_vector_1[1]-50])

        rand_placement_vector_2 = sample(range(50, 150), 2)
        while self.distance(rand_placement_vector_2[0], rand_placement_vector_2[1],
                            rand_placement_vector_1[0], rand_placement_vector_1[1]) < 20:
            rand_placement_vector_2 = sample(range(50, 150), 2)
        background.paste(_img_2, (rand_placement_vector_2[0]-50, rand_placement_vector_2[1]-50), _img_2)

        rand_placement_vector_3 = sample(range(50, 150), 2)
        while self.distance(rand_placement_vector_2[0], rand_placement_vector_2[1],
                            rand_placement_vector_3[0], rand_placement_vector_3[1]) < 20:
            rand_placement_vector_3 = sample(range(50, 150), 2)
        background.paste(_img_3, (rand_placement_vector_3[0]-50, rand_placement_vector_3[1]-50), _img_3)

        background.save("cnn_data/R_" + str(icon_id) + ".png", "PNG")

    def create_samples(self, save_directory):

        files = os.listdir(save_directory)
        for i in range(len(files)):
            if files[i] == ".DS_Store":
                files.pop(i)
                break

        icon_id = 0

        for i in range(int(self.n_samples/4)):

            img_1 = choice(files)
            files.remove(img_1)

            img_2 = choice(files)
            files.remove(img_2)

            img_3 = choice(files)
            files.remove(img_3)

            self.create_negative_sample(save_directory + "/" + img_1,
                                        save_directory + "/" + img_2,
                                        save_directory + "/" + img_3,
                                        icon_id)
            icon_id += 1

            os.remove(save_directory + "/" + img_1)
            os.remove(save_directory + "/" + img_2)
            os.remove(save_directory + "/" + img_3)

    def pickle_dataset(self):

        pickled_images = open('pkl_images.pkl', 'wb')
        pickled_labels = open('pkl_labels.pkl', 'wb')

        images = []
        labels = []

        for i in os.listdir("cnn_data"):
            current_image = "cnn_data/{}".format(i)
            _img = Image.open(current_image).convert('L')
            _img_arr = asarray(_img)

            images.append(_img_arr)

            if i[0].upper() == "I":
                labels.append(1)
            else:
                labels.append(0)

        pickle.dump(images, pickled_images)
        pickle.dump(labels, pickled_labels)

        shutil.rmtree("cnn_data")

    def load_pickled_dataset(self):
        pickled_images = open('pkl_images.pkl', 'rb')
        pickled_labels = open('pkl_labels.pkl', 'rb')

        images = pickle.load(pickled_images)
        labels = pickle.load(pickled_labels)

        _images = asarray(images)
        _labels = asarray(labels)

        return _images, _labels

    def generate_dataset(self, dataset_exists):
        if not dataset_exists:
            self.get_images("cnn_data")
            self.create_samples("cnn_data")
            self.pickle_dataset()
            data = self.load_pickled_dataset()
        else:
            data = self.load_pickled_dataset()

        # 60k n_samples: 15k of each class, 30k total

        split_point = int(0.8*(self.n_samples/2))

        images, labels = sklearn.utils.shuffle(data[0], data[1])

        images_train = images[0:split_point]
        labels_train = labels[0:split_point]

        images_test = images[split_point:]
        labels_test = labels[split_point:]

        print("Train images, labels: ", len(images_train),
              "\n Image array shape: ", images_train.shape,
              "\n Label array shape: ", labels_train.shape)

        print("Test images, labels: ", len(images_test),
              "\n Image array shape: ", images_test.shape,
              "\n Label array shape: ", labels_test.shape)

        return images_train, labels_train, images_test, labels_test

