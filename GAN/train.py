import os
import joblib
import pickletools
import gzip
from PIL import Image
import pickle
import sklearn
import numpy as np
from scipy import spatial
import random
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from models import Text2Image
import matplotlib.pyplot as plt


class TrainLib:
    def __init__(self, data_directory):
        self.glove_embeddings = np.load(data_directory + '/glove/glove_embeddings.npy', allow_pickle=True)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            models = Text2Image()
            self.discriminator = models.build_discriminator()
            self.generator = models.build_generator()
            self.combined = models.build_combined(generator=self.generator, discriminator=self.discriminator)

            self.discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                       optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                       metrics=['accuracy'])

            self.combined.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                  optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                  metrics=['accuracy'])

        self.EPOCHS = 20000
        self.BATCH_SIZE = 144
        self.NOISE_DIM = 100

        print("Loading images...")
        self.images = np.asarray(joblib.load(data_directory + '/full_images.compressed'))
        print("Loading labels...")
        self.labels = np.asarray(joblib.load(data_directory + '/full_labels.compressed'))

    def display_training_metrics(self, epoch, epochs, d_loss, g_loss):
        print("Epoch {}/{}".format(epoch, epochs))
        print("D loss: {:.4f}, acc: {:.4f} \nG loss: {:.4f}, acc {:.4f}".format(
            d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))

    def unnorm_image(self, image_array):
        return (image_array * 127.5) + 127.5

    def find_closest_embeddings(self, embedding):
        return sorted(self.glove_embeddings.item().keys(),
                      key=lambda word: spatial.distance.euclidean(self.glove_embeddings.item().get(word), embedding))

    def create_generator_samples(self, epoch, nrows=1, ncols=1):
        sampled_text_idx = np.random.randint(0, self.labels.shape[0], nrows*ncols)
        sampled_text = self.labels[sampled_text_idx]

        noise = self.generate_latent_points(nrows*ncols)
        generated_images = self.generator.predict([sampled_text, noise])

        unnorm_img = np.asarray(self.unnorm_image(generated_images[0])).astype(np.uint8)

        plt.figure()
        plt.imshow(np.reshape(unnorm_img, (200, 200)), cmap='gray')
        plt.title(self.find_closest_embeddings(sampled_text[0])[0])
        plt.savefig("training_samples/sample_{}.png".format(epoch), format="png")
        plt.close()

    def generate_latent_points(self, batch_size):
        x_latent = np.random.randn(batch_size * self.NOISE_DIM)
        x_latent = np.reshape(x_latent, (batch_size, self.NOISE_DIM))

        return x_latent

    def generate_real_samples(self, batch_size):
        idx = np.random.randint(0, self.images.shape[0], batch_size)

        x_image = self.images[idx]  # image with right caption
        x_text = self.labels[idx]
        y = np.ones((batch_size, 1))

        return x_image, x_text, y

    def generate_fake_samples(self, batch_size):
        idx = np.random.randint(0, self.labels.shape[0], batch_size)

        x_latent = self.generate_latent_points(batch_size)  # generated image w/ right caption
        x_text = self.labels[idx]

        x_generator = self.generator.predict([x_text, x_latent])
        y = np.zeros((batch_size, 1))

        return x_generator, x_text, y

    def generate_mismatch_samples(self, batch_size):
        idx_1 = np.random.randint(0, self.images.shape[0], batch_size)
        idx_2 = np.random.randint(0, self.labels.shape[0], batch_size)

        x_image = self.images[idx_1]  # image with wrong caption
        x_text = self.labels[idx_2]
        y = np.zeros((batch_size, 1))

        return x_image, x_text, y

    def train(self, sample_interval=250):
        valid = np.ones((self.BATCH_SIZE, 1))
        fake = np.zeros((self.BATCH_SIZE, 1))

        print("Beginning training...\n",
              tf.config.list_physical_devices('GPU'))

        batches_per_epoch = int(len(self.images) / self.BATCH_SIZE)
        n_steps = batches_per_epoch * self.BATCH_SIZE
        third_batch = int(n_steps / 3)

        for i in range(n_steps):
            # Generate real samples
            x_image_real, x_text_real, y_real = self.generate_real_samples(third_batch)  # real samples
            d_loss_1, d_acc_1 = self.discriminator.train_on_batch([x_image_real, x_text_real], y_real)

            # Generate fake samples
            x_image_fake, x_text_fake, y_fake = self.generate_fake_samples(third_batch)
            d_loss_2, d_acc_2 = self.discriminator.train_on_batch([x_image_fake, x_text_fake], y_fake)

            # Generate mismatch samples
            x_image_mismatch, x_text_mismatch, y_mismatch = self.generate_mismatch_samples(third_batch)
            d_loss_3, d_acc_3 = self.discriminator.train_on_batch([x_image_mismatch, x_text_mismatch], y_mismatch)

            # Train combined GAN model
            idx = np.random.randint(0, self.labels.shape[0], self.BATCH_SIZE)

            x_latent = self.generate_latent_points(self.BATCH_SIZE)
            x_text = self.labels[idx]
            y_gan = np.ones((self.BATCH_SIZE, 1))

            g_loss = self.combined.train_on_batch([x_text, x_latent], y_gan)

            if (i+1) % 50 == 0:
                print("batch {}:\n"
                      "d_loss_1 (real): {},  d_loss_2 (fake): {},  d_loss_3 (mismatch): {}\n"
                      "d_acc_1 (real): {},  d_acc_3 (fake): {},  d_acc_3 (mismatch): {}\n"
                      "g_loss: {}\n".format(i+1,
                                            d_loss_1, d_loss_2, d_loss_3,
                                            int(100*d_acc_1), int(100*d_acc_2), int(100*d_acc_3),
                                            g_loss))

            if (i+1) % 500 == 0:
                self.create_generator_samples(i+1)

        tf.keras.models.save_model(self.generator, "saved_text2icon_gan")


if __name__ == "__main__":
    Text2Im = TrainLib("/nethome/mreich8/VisualEssence/data/gan_data")
    Text2Im.train()
