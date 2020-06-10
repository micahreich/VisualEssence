import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import os
from random import randint, sample, random
import math
from IconGeneration import ConvexHull


class SearchPrediction:
    def __init__(self, data_directory, n_positions=1000):
        self.data_directory = data_directory
        self.n_positions = n_positions

    def predict(self, image_arr):
        model = tf.keras.models.load_model('saved_discriminator')
        image_arr = (255 - np.asarray(image_arr))[:, :, :, 2]

        return model.predict(image_arr / 255.0)

    def generate_pos_vec(self, icon_size=120):
        position_vector = []

        vec_1 = sample(range(60, 140), 2)  # (x, y)
        position_vector.append(vec_1)

        for i in range(2):
            radius = randint(50, 140)
            angle = random() * 6.28319  # 0 to 2pi radians

            while (position_vector[i][0] + radius * math.cos(angle) + 60 > 200 or position_vector[i][
                0] + radius * math.cos(angle) - 60 < 0) or \
                    (position_vector[i][1] + radius * math.sin(angle) + 60 > 200 or position_vector[i][
                        1] + radius * math.sin(angle) - 60 < 0):
                radius = randint(50, 140)
                angle = random() * 6.28319

            vec = [int(position_vector[i][0] + radius * math.cos(angle)),
                   int(position_vector[i][1] + radius * math.sin(angle))]
            position_vector.append(vec)

        return position_vector

    def create_samples(self):
        icons = sample(os.listdir(self.data_directory), 3)
        samples = []

        for i in range(self.n_positions):
            ConvexHull.convex_hull(self.data_directory,
                                   icons, self.generate_pos_vec(), i)
            samples.append(np.asarray(Image.open(self.data_directory + "/P_" + str(i) + ".png"))[:, :, 2])

        return np.reshape(samples, (len(samples), 200, 200, 1))

    def get_performance_samples(self):
        arrangements = self.create_samples()
        softmax_scores = self.predict(arrangements)[:, 1]  # class 1 scores

        positive_arrangement = Image.fromarray(arrangements[softmax_scores.index(max(softmax_scores))])
        negative_arrangement = Image.fromarray(arrangements[softmax_scores.index(min(softmax_scores))])

        positive_arrangement.save("Highest_Scoring_Sample.png", "PNG")
        negative_arrangement.save("Lowest_Scoring_Sample.png", "PNG")

        print("HIGHEST SCORE: ", max(softmax_scores))
        print("LOWEST SCORE: ", min(softmax_scores))


if __name__ == "__main__":
    SP = SearchPrediction("/nethome/mreich8/VisualEssence/data/cnn_data_backup_2/cnn_data")
    SP.get_performance_samples()
