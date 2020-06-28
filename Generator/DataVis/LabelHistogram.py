from PIL import Image
import tensorflow as tf
import numpy as np
import os
import numpy as np
import math
import matplotlib.mlab as mlab
import statistics
import pickle
import matplotlib.pyplot as plt
import random


class LabelHistogram:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.path_to_model = data_directory[:data_directory.find("VisualEssence")] + "VisualEssence/Generator/saved_pos_gen_45"

    def get_labels(self):
        pickled_labels = pickle.load(open((self.data_directory + '/all/pkl_labels_all.pkl'), 'rb'))

        x = np.asarray(pickled_labels)[:,:,0]
        y = np.asarray(pickled_labels)[:,:,1]

        return np.asarray(pickled_labels).flatten().tolist(), \
               np.asarray(x).flatten().tolist(), \
               np.asarray(y).flatten().tolist()

    def plot_histogram(self):
        all_coords, x, y, garbage = self.get_labels()

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

    def create_histogram(self, array, name):
        plt.hist(array, 15, alpha=0.5, label=name)
        plt.legend(loc='upper right')
        plt.savefig("histograms/" + name + ".png")
        plt.close()

    def create_subplot_hist(self, x1, y1, x2, y2, x3, y3):
        fig, axs = plt.subplots(2, 3, figsize=(11, 5))
        axs[0, 0].hist(x1, 15, alpha=0.5)
        axs[0, 0].set_title("x1")

        axs[0, 1].hist(x2, 15, alpha=0.5)
        axs[0, 1].set_title("x2")

        axs[0, 2].hist(x3, 15, alpha=0.5)
        axs[0, 2].set_title("x3")

        axs[1, 0].hist(y1, 15, alpha=0.5)
        axs[1, 0].set_title("y1")

        axs[1, 1].hist(y2, 15, alpha=0.5)
        axs[1, 1].set_title("y2")

        axs[1, 2].hist(y3, 15, alpha=0.5)
        axs[1, 2].set_title("y3")
        fig.tight_layout()

        plt.savefig("all_coords.png")
        plt.close()

    def get_test_set(self):
        pickled_images = pickle.load(open((self.data_directory + '/all/pkl_images_all.pkl'), 'rb'))
        pickled_labels = pickle.load(open((self.data_directory + '/all/pkl_labels_all.pkl'), 'rb'))

        stacked_images = []

        for triplet in pickled_images:
            stacked_images.append(np.dstack((triplet[0], triplet[1], triplet[2])))

        images = np.asarray(stacked_images) / 255.0
        labels = np.asarray(pickled_labels) / 200

        split_point = int(0.8 * len(images))

        self.x_train = images[0:split_point]
        self.y_train = labels[0:split_point]

        self.x_test = images[split_point:]
        self.y_test = labels[split_point:]

        self.x_train = np.reshape(np.asarray(self.x_train), (len(self.x_train), 1, 200, 200, 3))
        self.x_test = np.reshape(np.asarray(self.x_test), (len(self.x_test), 1, 200, 200, 3))

        self.y_train = np.reshape(self.y_train, newshape=(len(self.y_train), 6))
        self.y_test = np.reshape(self.y_test, newshape=(len(self.y_test), 6))

        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(len(self.x_train))
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).shuffle(len(self.x_test))

        return train_ds, test_ds

    def get_predicted_coords(self):
        self.get_test_set()
        model = tf.keras.models.load_model("/nethome/mreich8/VisualEssence/Generator/PG_2_Adam_3e4_tanh_1")
        predicted_coords = model.predict(self.x_test, batch_size=64)
        print(predicted_coords*200)
        return predicted_coords*200

    def plot_histogram_sep(self):
        _predictions = self.get_predicted_coords()
        predictions = np.reshape(_predictions, (-1, 3, 2))

        x1 = np.asarray(predictions)[:, :, 0][:, 0]
        y1 = np.asarray(predictions)[:, :, 1][:, 0]

        x2 = np.asarray(predictions)[:, :, 0][:, 1]
        y2 = np.asarray(predictions)[:, :, 1][:, 1]

        x3 = np.asarray(predictions)[:, :, 0][:, 2]
        y3 = np.asarray(predictions)[:, :, 1][:, 2]
        self.create_subplot_hist(x1, y1, x2, y2, x3, y3)

    def plot_metrics(self):
        hist = pickle.load(open("/nethome/mreich8/VisualEssence/Generator/PG_2_Adam_3e4_tanh_1.pkl", "rb"))
        plt.plot(hist['loss'], 'g', alpha=0.5, label='Training loss')
        plt.plot(hist['val_loss'], 'b', alpha=0.5, label='Validation loss')
        #plt.plot(hist['val_cosine_similarity'], 'r', alpha=0.5, label='Validation Cosine Similarity')
        #plt.plot(hist['cosine_similarity'], 'g', alpha=0.5, label='Training Cosine Similarity')
        plt.yscale('linear')
        plt.title('Loss (normalized, Adam 3e-4, tanh)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig("metrics_graph.png")

if __name__ == "__main__":
    # "/nethome/mreich8/VisualEssence/data/cnn_data_backup/cnn_data"
    DV = LabelHistogram("/nethome/mreich8/VisualEssence/data/generator_data/pkl")
    DV.plot_histogram_sep()
