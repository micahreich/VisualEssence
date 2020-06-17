import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pickle


class DataAggregator:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def aggregate_data(self):
        fnames = os.listdir(self.data_directory)
        for i in fnames:
            if not i.endswith(".pkl"):
                fnames.remove(i)

        pickled_images = open((self.data_directory + '/all/pkl_images_all.pkl'), 'wb')
        pickled_labels = open((self.data_directory + '/all/pkl_labels_all.pkl'), 'wb')

        images = []
        position_vector_labels = []

        for i in range(len(fnames)):
            if "images" in fnames[i]:
                _images = pickle.load(open((self.data_directory + "/" + fnames[i]), 'rb'))
                images += _images
            elif "labels" in fnames[i]:
                _labels = pickle.load(open((self.data_directory + "/" + fnames[i]), 'rb'))
                position_vector_labels += _labels

        print("AGGREGATED {} TRIPLETS WITH {} LABELS".format(len(images), len(position_vector_labels)))

        pickle.dump(images, pickled_images)
        pickle.dump(position_vector_labels, pickled_labels)


if __name__ == "__main__":
    DA = DataAggregator("/Users/micahreich/Documents/VisualEssence/data")
    DA.aggregate_data()
