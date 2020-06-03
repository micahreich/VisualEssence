import requests
from PIL import Image
import PIL
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
    def __init__(self, n_samples, save_directory):
        try:
            os.mkdir(save_directory)
        except FileExistsError:
            print("WARNING: directory cnn_data already exists")

        self.n_samples = n_samples

    def get_images(self, save_directory):
        print("\n STARTING ICON COLLECTION")

        img_ids = []
        icon_id = 0

        for i in range(self.n_samples):
            if (i + 1) % 100 == 0:
                print("Grabbed " + str(i + 1) + " images")

            rand_icon_id = randint(0, 3368879)

            while rand_icon_id in img_ids:
                rand_icon_id = randint(0, 3368879)

            img_ids.append(rand_icon_id)

            img_url = "https://static.thenounproject.com/png/{}-200.png".format(rand_icon_id)

            with open(os.path.abspath(save_directory + "/I_" + str(rand_icon_id) + ".png"), 'wb') as f:
                f.write(requests.get(img_url).content)
            f.close()

            icon_id += 1

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def create_negative_sample(self, path_to_img_1, path_to_img_2, path_to_img_3, icon_id):

        # Assuming 200 x 200 images rescaled to 140 x 140 to fit in different locations
        # within the new 200 x 200 random amalgamation
        try:
            positions = []
            background = Image.open(os.path.abspath("blank_icon.png"))
            _img_1_i = Image.open(os.path.abspath(path_to_img_1))
            _img_2_i = Image.open(os.path.abspath(path_to_img_2))
            _img_3_i = Image.open(os.path.abspath(path_to_img_3))
            _img_1 = _img_1_i.resize((100, 100), Image.ANTIALIAS)
            _img_2 = _img_2_i.resize((100, 100), Image.ANTIALIAS)
            _img_3 = _img_3_i.resize((100, 100), Image.ANTIALIAS)

            # Assuming 200 x 200 background
            rand_placement_vector_1 = sample(range(50, 150), 2)
            background.paste(_img_1, (rand_placement_vector_1[0], rand_placement_vector_1[1]), _img_1)
            positions.append([rand_placement_vector_1[0] - 50, rand_placement_vector_1[1] - 50])

            rand_placement_vector_2 = sample(range(50, 150), 2)
            while self.distance(rand_placement_vector_2[0], rand_placement_vector_2[1],
                                rand_placement_vector_1[0], rand_placement_vector_1[1]) < 20:
                rand_placement_vector_2 = sample(range(50, 150), 2)
            background.paste(_img_2, (rand_placement_vector_2[0] - 50, rand_placement_vector_2[1] - 50), _img_2)

            rand_placement_vector_3 = sample(range(50, 150), 2)
            while self.distance(rand_placement_vector_2[0], rand_placement_vector_2[1],
                                rand_placement_vector_3[0], rand_placement_vector_3[1]) < 20:
                rand_placement_vector_3 = sample(range(50, 150), 2)
            background.paste(_img_3, (rand_placement_vector_3[0] - 50, rand_placement_vector_3[1] - 50), _img_3)
            background.save("cnn_data/R_" + str(icon_id) + ".png", "PNG")
        except PIL.UnidentifiedImageError as e:
            print("IMAGE ERROR, COULD NOT FIND: ", e)
        except ValueError:
            print("CORRUPTED IMGs: ", path_to_img_1, path_to_img_2, path_to_img_3, icon_id)

    def create_samples(self, save_directory):
        print("\n STARTING NEGATIVE SAMPLE CREATION")

        files = os.listdir(os.path.abspath(save_directory))
        for i in range(len(files)):
            if files[i] == ".DS_Store":
                files.pop(i)
                break

        good_fcount = 0
        for i in files:
            if i[0].upper() == "I":
                good_fcount += 1
            else:
                files.remove(i)

        icon_id = 0
        print(good_fcount)

        for i in range(int(good_fcount / 4)):
            if (i + 1) % 100 == 0:
                print("Generated " + str(i + 1) + " negative samples")

            img_1 = choice(files)
            while not os.path.exists(os.path.abspath(save_directory + "/" + img_1)):
                img_1 = choice(files)

            files.remove(img_1)

            img_2 = choice(files)
            while not os.path.exists(os.path.abspath(save_directory + "/" + img_2)):
                img_2 = choice(files)

            files.remove(img_2)

            img_3 = choice(files)
            while not os.path.exists(os.path.abspath(save_directory + "/" + img_3)):
                img_3 = choice(files)

            files.remove(img_3)

            self.create_negative_sample(os.path.abspath(save_directory + "/" + img_1),
                                        os.path.abspath(save_directory + "/" + img_2),
                                        os.path.abspath(save_directory + "/" + img_3),
                                        icon_id)
            icon_id += 1

            os.remove(os.path.abspath(save_directory + "/" + img_1))
            os.remove(os.path.abspath(save_directory + "/" + img_2))
            os.remove(os.path.abspath(save_directory + "/" + img_3))

    def pickle_dataset(self):
        print("\n PICKLING DATASET")
        files = os.listdir(os.path.abspath("cnn_data"))
        i_cnt = 0
        r_cnt = 0
        for i in files:
            if i[0].upper() == "I":
                i_cnt += 1
            else:
                r_cnt += 1
        print("POS IMGS ", i_cnt)
        print("NEG IMGS ", r_cnt)

        pickled_images = open(os.path.abspath('pkl_images.pkl'), 'wb')
        pickled_labels = open(os.path.abspath('pkl_labels.pkl'), 'wb')

        images = []
        labels = []

        for i in os.listdir(os.path.abspath("cnn_data")):
            try:
                current_image = "cnn_data/{}".format(i)
                _img_1 = Image.open(os.path.abspath(current_image))
                _img = _img_1.convert('L')
                _img_arr = asarray(_img)

                images.append(_img_arr)
                if i[0].upper() == "I":
                    labels.append(1)
                else:
                    labels.append(0)
            except PIL.UnidentifiedImageError:
                print("COULD NOT FIND IMAGE, CONTINUING...")

        pickle.dump(images, pickled_images)
        pickle.dump(labels, pickled_labels)
        print(len(images))
        print(asarray(images).shape)

        # shutil.rmtree("cnn_data")

    def load_pickled_dataset(self):
        print("\n LOADING PICKLED DATASET")

        pickled_images = open(os.path.abspath('pkl_images.pkl'), 'rb')
        pickled_labels = open(os.path.abspath('pkl_labels.pkl'), 'rb')

        images = pickle.load(pickled_images)
        labels = pickle.load(pickled_labels)

        _images = asarray(images)
        _labels = asarray(labels)

        return _images, _labels

    def generate_dataset(self, dataset_exists):
        if not dataset_exists:
            self.get_images("cnn_data")
            print("\nICON COLLECTION COMPLETED SUCCESSFULLY")
            self.create_samples("cnn_data")
            print("NEGATIVE SAMPLE CREATION COMPLETED SUCCESSFULLY")
            self.pickle_dataset()
            print("CREATED PICKLED DATASET SUCCESSFULLY")
        else:
            # self.create_samples("cnn_data")
            # print("\nNEGATIVE SAMPLE CREATION COMPLETED SUCCESSFULLY")
            # self.pickle_dataset()
            # print("PICKLED DATASET COMPLETED SUCCESSFULLY")
            data = self.load_pickled_dataset()

        # 60k n_samples: 15k of each class, 30k total
        print("GENERATING DATA SPLIT")

        split_point = int(0.8 * (self.n_samples / 2))

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
