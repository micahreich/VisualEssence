import requests
from PIL import Image
from random import randint
from random import sample
from random import choice
import os
import numpy as np
from numpy import asarray
from numpy import ndarray
import pickle


class DatasetGenerator:
    def __init__(self, n_samples):
        try:
            os.mkdir("cnn_data")
        except FileExistsError:
            print("cnn_data already exists")

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

    def create_negative_sample(self, path_to_img_1, path_to_img_2, path_to_img_3, icon_id):

        # Assuming 200 x 200 images rescaled to 140 x 140 to fit in different locations
        # within the new 200 x 200 random amalgamation
        background = Image.open("blank_icon.png")
        _img_1 = Image.open(path_to_img_1).resize((100, 100), Image.ANTIALIAS)
        _img_2 = Image.open(path_to_img_2).resize((100, 100), Image.ANTIALIAS)
        _img_3 = Image.open(path_to_img_3).resize((100, 100), Image.ANTIALIAS)

        # Assuming 200 x 200 background
        rand_placement_vector = sample(range(0, 100), 2)
        background.paste(_img_1, (rand_placement_vector[0], rand_placement_vector[1]), _img_1)

        rand_placement_vector = sample(range(0, 100), 2)
        background.paste(_img_2, (rand_placement_vector[0], rand_placement_vector[1]), _img_2)

        rand_placement_vector = sample(range(0, 100), 2)
        background.paste(_img_3, (rand_placement_vector[0], rand_placement_vector[1]), _img_3)

        background.save("cnn_data/R_" + str(icon_id) + ".png", "PNG")

    def generate_dataset(self, save_directory):

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
            print("IMG PICK:" + str(_img_arr.shape) + i)

            images.append(_img_arr)

            if i[0].upper() == "I":
                labels.append(1)
            else:
                labels.append(0)

        pickle.dump(images, pickled_images)
        pickle.dump(labels, pickled_labels)

        #os.rmdir("cnn_data")

    def load_pickled_dataset(self):
        pickled_images = open('pkl_images.pkl', 'rb')
        pickled_labels = open('pkl_labels.pkl', 'rb')

        images = pickle.load(pickled_images)
        labels = pickle.load(pickled_labels)


if __name__ == "__main__":
    DG = DatasetGenerator(4)
    """DG.get_images("cnn_data")
    DG.generate_dataset("cnn_data")
    DG.pickle_dataset()
    DG.load_pickled_dataset()"""
