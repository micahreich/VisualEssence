from PIL import Image
import pickle
import numpy as np


def load_image(path_to_pkl):
    pickled_images_0 = open((path_to_pkl + '/pkl_images_0.pkl'), 'rb')
    pickled_images_1 = open((path_to_pkl + '/pkl_images_1.pkl'), 'rb')

    images_0 = pickle.load(pickled_images_0)
    images_1 = pickle.load(pickled_images_1)

    for i in images_1:
        for j in i:
            im = Image.fromarray(j).show()


def count(path_to_pkl):
    pickled_images_0 = open((path_to_pkl + '/pkl_images_0.pkl'), 'rb')
    pickled_labels_1 = open((path_to_pkl + '/pkl_labels_0.pkl'), 'rb')

    images_0 = pickle.load(pickled_images_0)
    labels_0 = pickle.load(pickled_labels_1)

    print(labels_0[1])


count("/Users/micahreich/Documents/VisualEssence/data")