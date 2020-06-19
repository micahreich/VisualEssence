from PIL import Image
from random import randint, sample, random, shuffle, choice
import os
import numpy as np
import pickle
import math
import tensorflow as tf
import sys
from IconGeneration import ConvexHull


class DatasetGenerator:
    """def __init__(self, n_samples, data_directory, run_mode, slurm_array_task_id=sys.argv[1]):
        self.n_samples = n_samples  # n_samples must be divisible by 3 * n_partitions
        self.data_directory = data_directory
        self.n_partitions = 20
        self.n_searches = 250
        self.run_mode = run_mode
        self.slurm_array_task_id = slurm_array_task_id"""

    def generate_pos_vec(self, icon_size=120, n_gutters=10, difference_factor=15):
        position_vector = np.zeros(shape=(3, 2)).astype(np.int)
        position_vector_gutters = np.zeros(shape=(3, 2)).astype(np.int)

        gutter_width = 200 / n_gutters
        for i in range(3):
            gutter_range_x = choice([i for i in range(2, n_gutters-1) if i not in position_vector_gutters[:,0]])
            gutter_range_y = choice([i for i in range(2, n_gutters-1) if i not in position_vector_gutters[:,1]])

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

    def generate_partition(self):
        files = os.listdir(self.data_directory)[:self.n_samples]

        file_triplets = []
        for i in range(0, int(len(files)/3)*3, 3):
            file_triplets.append(files[i:i + 3])

        dataset_partition = file_triplets[
            int(self.slurm_array_task_id)*int(len(file_triplets)/self.n_partitions):
            int(self.slurm_array_task_id)*int(len(file_triplets)/self.n_partitions) + int(len(file_triplets)/self.n_partitions)
        ]

        for i in range(len(dataset_partition)):
            for j in range(len(dataset_partition[i])):
                dataset_partition[i][j] = self.data_directory + "/" + dataset_partition[i][j]

        print("GENERATED {} IMAGE FILE TRIPLETS".format(len(dataset_partition)))

        return dataset_partition

    def generate_samples(self, icons):
        samples = []
        position_vectors = []

        for i in range(self.n_searches):
            position = self.generate_pos_vec()
            chull_image = ConvexHull.convex_hull(self.data_directory,
                                                 icons, position)

            samples.append(np.asarray(chull_image)[:, :, 2])
            position_vectors.append(position)
        print("GENERATED {} CONVEX HULL ARRANGEMENTS".format(len(samples)))
        print("GENERATED {} POSITION VECTORS".format(len(position_vectors)))

        return samples, position_vectors

    def highest_sample(self, arrangements):
        model = tf.keras.models.load_model(
            self.data_directory[:self.data_directory.index("VisualEssence")] + "VisualEssence/CNN/saved_discriminator"
        )

        softmax_scores = model.predict((np.reshape(arrangements, (-1, 200, 200, 1)) / 255.0), batch_size=50)[:, 1]
        idx_max = np.where(softmax_scores == np.amax(softmax_scores))[0][0]

        return idx_max

    def generate_dataset(self):
        partitions = self.generate_partition()
        images = []
        position_vector_labels = []

        pickled_images = open((self.data_directory + '/pkl_images' + str(self.slurm_array_task_id) + '.pkl'), 'wb')
        pickled_labels = open((self.data_directory + '/pkl_labels' + str(self.slurm_array_task_id) + '.pkl'), 'wb')

        for i in range(len(partitions)):
            if (i+1) % 50 == 0:
                print("GENERATED {} LABELS, {}% COMPLETE".format((i+1), int((i+1)/len(partitions))))

            samples, position_vectors = self.generate_samples(partitions[i])
            idx_max = self.highest_sample(samples)

            position_vector_labels.append(position_vectors[idx_max])

            image_triplet = []
            for j in range(len(partitions[i])):
                image_triplet.append((255 - np.asarray(Image.open(partitions[i][j])))[:, :, 3])

            images.append(image_triplet)
            tf.keras.backend.clear_session()

        print("GENERATED {} LABELS FOR {} IMAGE TRIPLETS".format(len(position_vector_labels), len(images)))

        pickle.dump(images, pickled_images)
        pickle.dump(position_vector_labels, pickled_labels)


if __name__ == "__main__":
    DS = DatasetGenerator().generate_pos_vec()
