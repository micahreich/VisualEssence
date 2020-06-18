from IconGeneration import ConvexHull
import tensorflow as tf
from PIL import Image
import random
import numpy as np
import os

class PositionGenEval:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.path_to_model = data_directory[data_directory.find("VisualEssence")] + "VisualEssence/Generator/saved_pos_gen"

    def create_sample(self, icons, position_vector):
        chull_image = ConvexHull.convex_hull(self.data_directory,
                                             icons, position_vector)
        Image.save(chull_image, "PosGenSample_{}".format(random.randint(0, 100)), "PNG")

    def predict(self, icon_paths):
        icons_array = []
        icons_dstack = []

        for i in range(len(icon_paths)):
            icon_triplet = []
            for j in range(len(icon_paths[i])):
                icon_triplet.append(255 - np.asarray(Image.open(icon_paths[i][j]))[:, :, 3])
            icons_array.append(icon_triplet)

        for i in range(len(icons_array)):
            icons_dstack.append(np.dstack((icons_array[i][0], icons_array[i][1], icons_array[i][2])))

        icons_dstack = np.asarray(icons_dstack).reshape(shape=(len(icons_dstack), 1, 200, 200, 3))

        icons_dstack /= 255.0

        prediction_vectors = []
        model = tf.keras.models.load_model(self.path_to_model)

        for i in range(len(icons_dstack)):
            tf.keras.backend.clear_session()
            prediction_vectors.append(
                model.predict(icons_dstack[i], batch_size=64)
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

        for i in range(len(icon_paths)):
            for j in range(len(icon_paths[i])):
                icon_paths[i][j] = self.data_directory + "/" + icon_paths[i][j]

        for i in range(n_samples):
            self.create_sample(
                icons=icon_paths[i], position_vector=np.reshape(self.predict(icon_paths[i]), newshape=(3, 2))
            )
