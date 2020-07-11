import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *


class EvalLib:
    def __init__(self, save_directory):
        self.n_classes = 10
        self.latent_dim = 100

        self.model = tf.keras.models.load_model(save_directory)

    def gen_combined_label(self, active_classes):
        combined_label = [0]*self.n_classes

        for label in active_classes:
            combined_label[label] = 1

        return np.reshape(np.asarray(combined_label), (1, self.n_classes))

    def sample_combined_inputs(self):
        noise = np.random.uniform(-1.0, 1.0, size=[1, self.latent_dim])
        combined_label = self.gen_combined_label([0, 1])

        gen_img = self.model.predict([noise, combined_label])

        plt.imshow(gen_img[0], cmap='gray')
        plt.title(str(list(combined_label)).strip('[]'))
        plt.axis('off')
        plt.savefig("eval_images/combined.png", bbox_inches='tight', dpi=200)
        plt.close()


if __name__ == "__main__":
    EL = EvalLib('mnist_generator/')
    EL.sample_combined_inputs()
