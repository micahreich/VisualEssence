from PIL import Image
import numpy as np
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random


class PixelHistogram:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def count_black_px(self, path_to_image, pos_neg_sample):
        black_px_count = 0
        image_arr = np.asarray(Image.open(path_to_image))

        if pos_neg_sample == "POSITIVE":
            image_arr = (255-image_arr)[:, :, 3]
        elif pos_neg_sample == "NEGATIVE":
            image_arr = image_arr[:, :, 2]

        return np.count_nonzero(image_arr < 255)

    def get_pixel_data(self):
        positive_sample_px = []
        negative_sample_px = []

        files = os.listdir(self.data_directory)

        for i in files:
            if i[0].upper() == "I":
                positive_sample_px.append(self.count_black_px(self.data_directory + "/" + i, "POSITIVE"))
            elif i[0].upper() == "R":
                negative_sample_px.append(self.count_black_px(self.data_directory + "/" + i, "NEGATIVE"))

        return positive_sample_px, negative_sample_px

    def plot_histogram(self):
        pixel_counts = self.get_pixel_data()

        plt.hist(pixel_counts[0], 10, alpha=0.5, label='Positive Samples')
        plt.hist(pixel_counts[1], 10, alpha=0.5, label='Negative Samples')
        plt.legend(loc='upper right')
        plt.savefig("PixelHistogram.png")


if __name__ == "__main__":
    # "/nethome/mreich8/VisualEssence/data/cnn_data_backup/cnn_data"
    DataVisualizer = PixelHistogram("/Users/micahreich/Documents/VisualEssence/data/cnn_data")
    DataVisualizer.plot_histogram()
