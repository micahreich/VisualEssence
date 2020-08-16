import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import random
from models import ModelLib
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


class TrainLib:
    def __init__(self):
        self.epochs = 12000
        self.batch_size = 64
        self.img_size = 72
        self.img_channels = 3
        self.latent_dim = 100

        models = ModelLib()

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(0.0003, 0.5)

            self.discriminator = models.build_discriminator()
            self.discriminator.compile(
                loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.4),
                optimizer=optimizer,
                metrics=['accuracy'])

            self.composer = models.build_composer()

            self.gan = models.build_full_model(composer=self.composer, discriminator=self.discriminator)
            self.gan.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=optimizer)

        print("Loading Squares dataset...")
        self.x_train = np.load("data/squares.npy", allow_pickle=True)

        print(self.x_train.shape)

    def generate_latent_noise(self, batch_size):
        return np.random.randn(batch_size * self.latent_dim).reshape((batch_size, self.latent_dim))

    def generate_real_samples(self, batch_size):
        idx = np.random.randint(0, self.x_train.shape[0], batch_size)
        return self.x_train[idx]

    def noisy_labels(self, y, p_flip):
        n_select = int(p_flip * y.shape[0])
        flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
        y[flip_ix] = 1 - y[flip_ix]
        return y

    def sample_images(self, epoch):
        r, c = 4, 4

        noise = self.generate_latent_noise(batch_size=r*c)
        gen_imgs = self.composer.predict(noise) / 255.0

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def train(self, sample_interval=100):
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            #  Train discriminator
            x_real = self.generate_real_samples(self.batch_size)
            noise = self.generate_latent_noise(self.batch_size)

            gen_imgs = self.composer.predict(noise, batch_size=self.batch_size)

            d_loss_real = self.discriminator.train_on_batch(x_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = self.generate_latent_noise(self.batch_size)
            g_loss = self.gan.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.composer.save('square_gen')


if __name__ == "__main__":
    TL = TrainLib()
    TL.train()
