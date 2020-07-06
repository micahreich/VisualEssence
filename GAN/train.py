import os
from PIL import Image
import pickle
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
        self.pickled_wv = dict(pickle.load(open('/nethome/mreich8/VisualEssence/data/gan_data/glove/glove_300d.pkl', 'rb')))

        self.discriminator = Text2Image.build_discriminator()
        self.generator = Text2Image.build_generator()
        self.combined = Text2Image.build_combined(generator=self.generator, discriminator=self.discriminator)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                                       metrics=['accuracy'])

            self.combined.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                                  metrics=['accuracy'])

        self.EPOCHS = 20000
        self.BATCH_SIZE = 64
        self.NOISE_DIM = 100

        print("Unpickling dataset...")
        self.train_data = np.asarray(pickle.load(open(data_directory, 'rb')))

    def discriminator_loss(self, real_loss, mismatch_loss, fake_loss):
        loss_array = [real_loss, mismatch_loss, fake_loss]
        total_loss = 0

        for i in loss_array:
            total_loss = np.add(total_loss, i)

        return 0.333 * total_loss

    def display_training_metrics(self, epoch, epochs, d_loss, g_loss):
        print("Epoch {}/{}".format(epoch, epochs))
        print("D loss: {:.4f}, acc: {:.4f} \nG loss: {:.4f}".format(
            d_loss[0], 100*d_loss[1], g_loss))

    def unnorm_image(self, image_array):
        return (image_array * 127.5) + 127.5

    def find_closest_embeddings(self, embedding):
        return sorted(self.pickled_wv.keys(),
                      key=lambda word: spatial.distance.euclidean(self.pickled_wv[word], embedding))

    def create_generator_samples(self, epoch, nrows=2, ncols=2):
        sampled_text_idx = np.random.randint(0, self.train_data.shape[0], nrows*ncols)
        sampled_text = self.train_data[sampled_text_idx, 1]

        noise = np.random.normal(0, 1, (nrows*ncols, self.NOISE_DIM))

        generated_images = self.generator.predict([sampled_text, noise])

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        for i in range(nrows):
            for j in range(ncols):
                img = Image.fromarray(self.unnorm_image(generated_images[i*j]))
                word = self.find_closest_embeddings(sampled_text[i*j])[0]

                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].set_title("Icon: {}".format(word))
                axs[i, j].axis('off')

        fig.subplots_adjust(wspace=0, hspace=0.3)
        fig.savefig(os.getcwd() + "/training_samples/sample_{}.png".format(epoch))

    @tf.function
    def train(self, sample_interval=100):
        valid = np.ones((self.BATCH_SIZE, 1))
        fake = np.zeros((self.BATCH_SIZE, 1))

        print("Beginning training...")
        for epoch in range(self.EPOCHS):
            #  Train Discriminator
            real_image_text_pairs_idx = np.random.randint(0, self.train_data.shape[0], self.BATCH_SIZE)
            real_images, real_text = self.train_data[real_image_text_pairs_idx, 0], \
                                     self.train_data[real_image_text_pairs_idx, 1]

            fake_text_idx = np.random.randint(0, self.train_data.shape[0], self.BATCH_SIZE)
            fake_text = self.train_data[fake_text_idx, 1]

            noise = np.random.normal(0, 1, (self.BATCH_SIZE, self.NOISE_DIM))
            generated_images = self.generator.predict([fake_text, noise])

            d_loss_real = self.discriminator.train_on_batch([real_images, real_text], valid)  # real image, real caption
            d_loss_mismatched = self.discriminator.train_on_batch([real_images, fake_text], fake)  # real image, mismatched caption
            d_loss_fake = self.discriminator.train_on_batch([generated_images, fake_text], fake)  # fake image, real caption

            d_loss = self.discriminator_loss(d_loss_real, d_loss_mismatched, d_loss_fake)

            #  Train Generator
            sampled_text_idx = np.random.randint(0, self.train_data.shape[0], self.BATCH_SIZE)
            sampled_text = self.train_data[sampled_text_idx, 1]

            noise = np.random.normal(0, 1, (self.BATCH_SIZE, self.NOISE_DIM))
            g_loss = self.combined.train_on_batch([sampled_text, noise])

            self.display_training_metrics(epoch=epoch, epochs=self.EPOCHS, d_loss=d_loss, g_loss=g_loss)

            if epoch % sample_interval == 0:
                print("Creating generator samples...")
                self.create_generator_samples(epoch)

        tf.keras.models.save_model(self.generator, "saved_text2icon_gan")


if __name__ == "__main__":
    Text2Im = TrainLib("/nethome/mreich8/VisualEssence/data/gan_data/all_norm/")
    Text2Im.train()
