from PIL import Image
import numpy as np
import os
import numpy as np
import math
import matplotlib.mlab as mlab
import pickle
import matplotlib.pyplot as plt
import random


class LabelHistogram:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def get_labels(self):
        pickled_labels = pickle.load(open((self.data_directory + '/all/pkl_labels_all.pkl'), 'rb'))

        x = np.asarray(pickled_labels)[:,:,0]
        y = np.asarray(pickled_labels)[:,:,1]

        return np.asarray(pickled_labels).flatten().tolist(), \
               np.asarray(x).flatten().tolist(), \
               np.asarray(y).flatten().tolist()

    def plot_histogram(self):
        all_coords, x, y = self.get_labels()

        bin_size_1 = 1 + 3.322*math.log(len(all_coords), 10)
        bin_size_2 = 1 + 3.322*math.log(len(x), 10)

        plt.hist(all_coords, int(bin_size_1), alpha=0.5, label='All Coordinates')
        plt.legend(loc='upper right')
        plt.savefig("AllCoords.png")
        plt.close()

        plt.hist(x, int(bin_size_2), alpha=0.5, label='X')
        plt.hist(y, int(bin_size_2), alpha=0.5, label='Y')
        plt.legend(loc='upper right')
        plt.savefig("XYCoords.png")


if __name__ == "__main__":
    # "/nethome/mreich8/VisualEssence/data/cnn_data_backup/cnn_data"
    DV = LabelHistogram("/Users/micahreich/Documents/VisualEssence/data")
    DV.plot_histogram()
