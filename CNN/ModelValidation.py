import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import os
from random import randint, sample, random, shuffle, choice
import math
from IconGeneration import ConvexHull


class SearchPrediction:
    def __init__(self, data_directory, n_positions=250):
        self.data_directory = data_directory
        self.n_positions = n_positions

    def predict(self, image_arr):
        model = tf.keras.models.load_model('saved_discriminator')
        image_arr = np.asarray(image_arr)

        return model.predict(image_arr / 255.0)

    def generate_pos_vec(self, icon_size=120, n_gutters=10, difference_factor=15):
        position_vector = np.zeros(shape=(3, 2)).astype(np.int)
        position_vector_gutters = np.zeros(shape=(3, 2)).astype(np.int)

        gutter_width = 200 / n_gutters
        for i in range(3):
            gutter_range_x = choice([i for i in range(2, n_gutters - 1) if i not in position_vector_gutters[:, 0]])
            gutter_range_y = choice([i for i in range(2, n_gutters - 1) if i not in position_vector_gutters[:, 1]])

            random_variance_x = randint(-difference_factor, difference_factor)
            random_variance_y = randint(-difference_factor, difference_factor)

            random_gutter_coord = [
                int((gutter_range_x * gutter_width) - (gutter_width / 2)) + random_variance_x,
                int((gutter_range_y * gutter_width) - (gutter_width / 2)) + random_variance_y
            ]

            position_vector[i] = random_gutter_coord
            position_vector_gutters[i] = [gutter_range_x, gutter_range_y]

        position_vector = position_vector.astype(int).tolist()

        return position_vector

    def generate_samples(self):
        icons = sample(os.listdir(self.data_directory), 3)
        samples = []

        for i in range(self.n_positions):
            ConvexHull.convex_hull(self.data_directory,
                                   icons, self.generate_pos_vec(), i)
            samples.append(np.asarray(Image.open(self.data_directory + "/P_" + str(i) + ".png"))[:, :, 2])

        return samples

    def get_performance_samples(self):
        model = tf.keras.models.load_model('saved_discriminator')

        arrangements = self.generate_samples()
        softmax_scores = model.predict((np.reshape(arrangements, (-1, 200, 200, 1))/255.0), batch_size=64)[:, 1]  # class 1 scores

        idx_max = np.where(softmax_scores == np.amax(softmax_scores))[0][0]
        idx_min = np.where(softmax_scores == np.amin(softmax_scores))[0][0]

        positive_arrangement = Image.fromarray(arrangements[idx_max])
        negative_arrangement = Image.fromarray(arrangements[idx_min])

        positive_arrangement.save("Highest_Scoring_Sample.png", "PNG")
        negative_arrangement.save("Lowest_Scoring_Sample.png", "PNG")

        print("HIGHEST SCORE: ", softmax_scores[idx_max])
        print("LOWEST SCORE: ", softmax_scores[idx_min])


if __name__ == "__main__":
    SP = SearchPrediction("/nethome/mreich8/VisualEssence/data/cnn_data_backup_2/cnn_data")
    SP.get_performance_samples()
