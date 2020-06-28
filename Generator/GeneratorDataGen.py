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
    def __init__(self, n_samples, data_directory, run_mode, slurm_array_task_id=sys.argv[1]):
        self.n_samples = n_samples  # n_samples must be divisible by 3 * n_partitions
        self.data_directory = data_directory
        self.n_partitions = 1
        self.n_searches = 230
        self.run_mode = run_mode
        self.slurm_array_task_id = slurm_array_task_id

        self.model = tf.keras.models.load_model(
            data_directory[:data_directory.index("VisualEssence")] + "VisualEssence/CNN/saved_discriminator"
        )

    def generate_pos_vec(self, threshold=60):
        position_vector = [sample(range(50, 150), 2)]

        for i in range(2):
            random_coord = sample(range(50, 150), 2)
            if i == 1:
                while self.distance(random_coord, position_vector[0]) <= threshold or \
                        self.distance(random_coord, position_vector[1]) <= threshold:
                    random_coord = sample(range(50, 150), 2)
                position_vector.append(random_coord)
            else:
                while self.distance(random_coord, position_vector[0]) <= threshold:
                    random_coord = sample(range(50, 150), 2)
                position_vector.append(random_coord)

        return position_vector

    def generate_partition(self):
        files = os.listdir(self.data_directory)[:self.n_samples]

        for i in files:
           if (not i.endswith(".png")) or (not i.startswith("I")):
               files.remove(i)

        file_triplets = []

        for i in range(0, int(len(files)/3)*3, 3):
            file_triplets.append(files[i:i + 3])

        dataset_partition = file_triplets[
            int(self.slurm_array_task_id)*int(len(file_triplets)/self.n_partitions):
            int(self.slurm_array_task_id)*int(len(file_triplets)/self.n_partitions) + int(len(file_triplets)/self.n_partitions)
        ]

        #for i in range(len(dataset_partition)):
        #    for j in range(len(dataset_partition[i])):
        #        dataset_partition[i][j] = self.data_directory + "/" + dataset_partition[i][j]

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
        softmax_scores = self.model.predict((np.reshape(arrangements, (-1, 200, 200, 1)) / 255.0), batch_size=50)[:, 1]
        idx_max = np.where(softmax_scores == np.amax(softmax_scores))[0][0]

        return idx_max

    def generate_dataset(self):
        partitions = self.generate_partition()
        images = []
        position_vector_labels = []

        pickled_images = open((self.data_directory + '/pkl_images_2_' + str(self.slurm_array_task_id) + '.pkl'), 'wb')
        pickled_labels = open((self.data_directory + '/pkl_labels_2_' + str(self.slurm_array_task_id) + '.pkl'), 'wb')

        for i in range(len(partitions)):
            if (i+1) % 50 == 0:
                print("GENERATED {} LABELS, {}% COMPLETE".format((i+1), int((i+1)/len(partitions))))

            samples, position_vectors = self.generate_samples(partitions[i])
            idx_max = self.highest_sample(samples)

            position_vector_labels.append(position_vectors[idx_max])
            print(position_vectors[idx_max])

            image_triplet = []
            for j in range(len(partitions[i])):
                image_triplet.append((255 - np.asarray(Image.open(self.data_directory + "/" + partitions[i][j])))[:, :, 3])

            images.append(image_triplet)
            tf.keras.backend.clear_session()

        print("GENERATED {} LABELS FOR {} IMAGE TRIPLETS".format(len(position_vector_labels), len(images)))

        pickle.dump(images, pickled_images)
        pickle.dump(position_vector_labels, pickled_labels)


if __name__ == "__main__":
    DS = DatasetGenerator(n_samples=150, data_directory="/nethome/mreich8/VisualEssence/data/generator_data", run_mode="?").generate_dataset()
