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
                                       optimizer=tf.keras.optimizers.Adam(0.0005, 0.5),
                                       metrics=['accuracy'])

            self.combined.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                  optimizer=tf.keras.optimizers.Adam(0.0005, 0.5),
                                  metrics=['accuracy'])

        self.EPOCHS = 20000
        self.BATCH_SIZE = 32
        self.NOISE_DIM = 100

        #print("Loading dataset...")
        #self.images = np.load(data_directory + '/full_dataset_images.npy', allow_pickle=True)
        #self.labels = np.load(data_directory + '/full_dataset_labels.npy', allow_pickle=True)
        print("Loading images...")
        self.images = np.asarray(joblib.load(data_directory + '/full_images.compressed'))
        print("Loading labels...")
        self.labels = np.asarray(joblib.load(data_directory + '/full_labels.compressed'))
        #print(self.images.shape, self.labels.shape)
        #self.images = np.asarray(pickle.load(open(data_directory + '/full_images.pickle', 'rb')))
        #self.labels = np.asarray(pickle.load(open(data_directory + '/full_labels.pickle', 'rb')))
        
    def discriminator_loss(self, real_loss, mismatch_loss, fake_loss):
        loss_array = [real_loss, mismatch_loss, fake_loss]
        total_loss = 0

        for i in loss_array:
            total_loss = np.add(total_loss, i)

        return 0.333 * total_loss

    def display_training_metrics(self, epoch, epochs, d_loss, g_loss):
        print("Epoch {}/{}".format(epoch, epochs))
        #print(d_loss, g_loss)
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

        noise = np.random.normal(0, 1, (nrows*ncols, self.NOISE_DIM))
        generated_images = self.generator.predict([sampled_text, noise])

        unnorm_img = np.asarray(self.unnorm_image(generated_images[0])).astype(np.uint8)

        plt.figure()
        plt.imshow(np.reshape(unnorm_img, (200, 200)), cmap='gray')
        plt.title(self.find_closest_embeddings(sampled_text[0])[0])
        plt.savefig("training_samples/sample_{}.png".format(epoch), format="png")
        plt.close()

    def train(self, sample_interval=250):
        valid = np.ones((self.BATCH_SIZE, 1))
        fake = np.zeros((self.BATCH_SIZE, 1))

        print("Beginning training...\n",
              tf.config.list_physical_devices('GPU'))

        batches_per_epoch = int(len(self.images) / self.BATCH_SIZE)
        n_steps = batches_per_epoch * self.BATCH_SIZE
        half_batch = int(n_steps / 2)

        for epoch in range(n_steps):
            #  Train Discriminator
            real_image_text_pairs_idx = np.random.randint(0, self.images.shape[0], self.BATCH_SIZE)
            real_images, real_text = np.reshape(self.images[real_image_text_pairs_idx], (-1, 200, 200, 1)), \
                                     np.reshape(self.labels[real_image_text_pairs_idx], (-1, 300))

            fake_text_idx = np.random.randint(0, self.labels.shape[0], self.BATCH_SIZE)
            fake_text = np.reshape(self.labels[fake_text_idx], (-1, 300))

            noise = np.random.normal(0, 1, (self.BATCH_SIZE, self.NOISE_DIM))
            generated_images = self.generator.predict([fake_text, noise])

            d_loss_real = self.discriminator.train_on_batch([real_images, real_text], valid)  # real image, real caption
            d_loss_mismatched = self.discriminator.train_on_batch([real_images, fake_text], fake)  # real image, mismatched caption
            d_loss_fake = self.discriminator.train_on_batch([generated_images, fake_text], fake)  # fake image, real caption

            d_loss = self.discriminator_loss(d_loss_real, d_loss_mismatched, d_loss_fake)

            #  Train Generator
            sampled_text_idx = np.random.randint(0, self.labels.shape[0], self.BATCH_SIZE)
            sampled_text = np.reshape(self.labels[sampled_text_idx], (-1, 300))

            noise = np.random.normal(0, 1, (self.BATCH_SIZE, self.NOISE_DIM))
            g_loss = self.combined.train_on_batch([sampled_text, noise], valid)

            if epoch % sample_interval == 0:
                self.display_training_metrics(epoch=epoch, epochs=self.EPOCHS, d_loss=d_loss, g_loss=g_loss)
                print("Creating generator samples...")
                self.create_generator_samples(epoch)

        tf.keras.models.save_model(self.generator, "saved_text2icon_gan")


if __name__ == "__main__":
    Text2Im = TrainLib("/nethome/mreich8/VisualEssence/data/gan_data")
    Text2Im.train()
