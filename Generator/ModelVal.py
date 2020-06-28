from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import tensorflow as tf
from random import sample
import os
import pickle
from PIL import Image
from IconGeneration import ConvexHull


class ModelVal:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.path_to_model = data_directory[:data_directory.find("VisualEssence")] + "VisualEssence/Generator/resnet_models/PG_ResNet_2_Adam_3e4_relu"

        self.fnames = os.listdir(data_directory)

        for i in self.fnames:
            if (not i.endswith(".png")) or (not i.startswith("I")):
                self.fnames.remove(i)
        self.triplets = triplets = [
            ['I_351320.png', 'I_966815.png', 'I_333188.png'],
            ['I_203141.png', 'I_2724271.png', 'I_2738218.png'],
            ['I_1723953.png', 'I_1840970.png', 'I_1747683.png']
        ]
        self.x_train=None
        self.y_train=None
        self.x_test=None
        self.y_test=None

    def get_random_triplet(self):
        return sample(population=self.fnames, k=3)

    def predict_triplet(self, stacked_input):
        model = tf.keras.models.load_model(self.path_to_model)

        return model.predict(stacked_input)

    def get_test_data(self):
        pickled_images = pickle.load(open((self.data_directory + '/pkl/all/pkl_images_all.pkl'), 'rb'))

        stacked_images = []

        for triplet in pickled_images:
            stacked_images.append(np.dstack((triplet[0], triplet[1], triplet[2])))

        images = np.asarray(stacked_images) / 255.0

        split_point = int(0.8 * len(images))

        self.x_test = images[split_point:]

    def center_samples(self):
      
        _id = 0

        for random_triplet in triplets:
            #_id+=1
            #x_input = np.random.randn(10*1).reshape(1, 10)
            #position =
            ConvexHull.convex_hull(self.data_directory, random_triplet, [[100, 100], [100, 100], [100, 100]]).save("center_samples/sample_arrangement_" + str(_id) + ".png", "PNG")

    def samples_by_trainset(self):
        self.get_test_data()
        model = tf.keras.models.load_model("/nethome/mreich8/VisualEssence/Generator/resnet_models/PG_ResNet_2_Adam_3e4_relu")
        predicted_coords = model.predict(self.x_test.reshape(-1, 1, 200, 200, 3), batch_size=64).astype(np.int).reshape(-1, 3, 2)
        _id = 0
        for random_triplet in self.triplets:
            _id+=1
            sampled_pos = random.sample(list(predicted_coords), 1)
            ConvexHull.convex_hull(self.data_directory, random_triplet, sampled_pos[0]).save("test_samples/sample_arrangement_" + str(_id) + ".png", "PNG")
    def samples_by_gan(self):
       model = tf.keras.models.load_model("/nethome/mreich8/VisualEssence/GAN/generator_GAN_nois2pos")
       _id = 0
       for random_triplet in self.triplets:
           x_input = np.random.randn(10*1).reshape(1, 10)
           _id += 1
           position_pred = model.predict(x_input).astype(np.int).reshape(-1, 3, 2)
           print(position_pred)
           ConvexHull.convex_hull(self.data_directory, random_triplet, list(position_pred[0])).save("test_samples/sample_arrangement_" + str(_id) + ".png", "PNG")

    def get_sample(self):

        _id = 0

        for random_triplet in self.triplets:
            _id+=1
            random_triplet_gray = []

            for i in range(len(random_triplet)):
                random_triplet_gray.append(
                    (255 - np.asarray(Image.open(self.data_directory + "/" + random_triplet[i])))[:, :, 3]
                )

            stacked_input = np.dstack((random_triplet_gray[0], random_triplet_gray[1], random_triplet_gray[2]))

            position = self.predict_triplet(
                np.reshape(stacked_input / 255.0, (1, 1, 200, 200, 3))
            ).astype(np.int)
        
            position = np.reshape(position, (-1, 3, 2)).tolist()
            print(position)

            ConvexHull.convex_hull(self.data_directory, random_triplet, position[0]).save("model_samples/sample_arrangement_" + str(_id) + ".png", "PNG")


if __name__ == "__main__":
    MV = ModelVal("/nethome/mreich8/VisualEssence/data/generator_data")
    MV.samples_by_gan()
    #for i in range(3): print(MV.get_random_triplet())

