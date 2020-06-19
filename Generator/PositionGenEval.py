from IconGeneration import ConvexHull
import tensorflow as tf
from PIL import Image
import random
import numpy as np
import os

class PositionGenEval:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.path_to_model = data_directory[:data_directory.find("VisualEssence")] + "VisualEssence/Generator/saved_pos_gen_2"

    def create_sample(self, icons, position_vector):
        chull_image = ConvexHull.convex_hull(self.data_directory,
                                             icons, position_vector)
        chull_image.save("PosGenSample_{}.png".format(random.randint(0, 100)), "PNG")

    def predict(self, icon_paths):
        icons_array = []
        icons_dstack = []

        icon_triplet = []
        for i in range(len(icon_paths)):
            icon_triplet.append(255 - np.asarray(Image.open(self.data_directory + "/" + icon_paths[i]))[:, :, 3])
        icons_array.append(icon_triplet)

        for i in range(len(icons_array)):
            icons_dstack.append(np.dstack((icons_array[i][0], icons_array[i][1], icons_array[i][2])))

        icons_dstack = np.reshape(np.asarray(icons_dstack), (len(icons_dstack), 1, 200, 200, 3))

        icons_dstack = np.asarray(icons_dstack) / 255.0

        prediction_vectors = []
        model = tf.keras.models.load_model(self.path_to_model)

        for i in range(len(icons_dstack)):
            prediction_vectors.append(
                model.predict(icons_dstack, batch_size=64)
            )

        return prediction_vectors

    def generate_samples(self, n_samples=1):
        data_path = os.listdir(self.data_directory)
        icon_paths = []

        for i in data_path:
            if not i.endswith(".png"):
                data_path.remove(i)

        for i in range(n_samples):
            icon_triplet = random.sample(population=data_path, k=3)
            icon_paths.append(icon_triplet)

            for fpath in icon_triplet:
                data_path.remove(fpath)

        for i in range(n_samples):
            pred = self.predict(icon_paths[i])
            print(pred)
            pred_reshape = (np.reshape(self.predict(icon_paths[i]), newshape=(3, 2))).astype(np.int)
            print(pred_reshape*200)
            self.create_sample(
                icons=icon_paths[i], position_vector=pred_reshape
            )


if __name__ == "__main__":
    PE = PositionGenEval("/nethome/mreich8/VisualEssence/data/generator_data")
    PE.generate_samples()
