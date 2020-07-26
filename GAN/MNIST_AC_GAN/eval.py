import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.layers import *


class EvalLib:
    def __init__(self, save_directory):
        self.num_classes = 10
        self.latent_dim = 100
        self.generator = tf.keras.models.load_model(save_directory)

    def generate_latent_noise(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.latent_dim))

    def generate_multi_hot_labels(self, y):
        multi_hot = []
        for label in y:
            multi_hot_label = [0] * self.num_classes
            for digit in label:
                multi_hot_label[digit] = 1
            multi_hot.append(multi_hot_label)

        return np.asarray(multi_hot)

    def sample_images(self):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        random_digits = np.random.randint(10, size=(r*c, 2))
        sampled_labels = self.generate_multi_hot_labels(y=random_digits)
        assert sampled_labels.shape == (r*c, 10), "ERR:" + str(sampled_labels.shape)

        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title(str(random_digits[cnt]), fontsize=8)
                cnt += 1
        fig.subplots_adjust(hspace=0.8)
        fig.savefig("samples.png")
        plt.close()

    def plot_training_history(self):
        history = pickle.load(open('training_history.pkl', 'rb'))
        d_loss_real = np.asarrat(history['d_loss_real'])
        d_loss_fake = np.asarrat(history['d_loss_fake'])

        d_loss = 0.5 * np.add(d_loss_real[:, 0], d_loss_fake[:, 0])
        g_loss = np.asarrat(history['g_loss'])

        plt.plot(d_loss, len(d_loss))
        plt.plot(g_loss, len(g_loss))

        plt.ylabel('Training Loss')
        plt.xlabel('Epoch')

        plt.savefig('training_loss.png')
        plt.close()


if __name__ == "__main__":
    EL = EvalLib('ac_gan_mnist/')
    EL.sample_images()

