import tensorflow as tf
import numpy as np
import pickle
import os
import random


class PositionGAN:
    def __init__(self, data_directory):
        self.pickled_labels = np.asarray(
            pickle.load(open((data_directory + '/all/pkl_labels_all.pkl'), 'rb'))
        ).reshape(-1, 6)

        self.strat = tf.distribute.MirroredStrategy()

        self.d_model = self.define_discriminator()
        self.g_model = self.define_generator(latent_dim=10)

        self.gan_model = self.define_gan(self.g_model, self.d_model)

    def define_discriminator(self, n_inputs=6):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=n_inputs))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          metrics=['accuracy'])

        return model

    def define_generator(self, latent_dim, n_outputs=6):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=latent_dim))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(n_outputs, activation='relu'))

        return model

    def define_gan(self, generator, discriminator):
        discriminator.trainable = False
        model = tf.keras.Sequential()
        model.add(generator)
        model.add(discriminator)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        return model

    def generate_latent_points(self, latent_dim, n):
        x_input = np.random.randn(latent_dim*n).reshape(n, latent_dim)
        return x_input

    def generate_fake_samples(self, generator, latent_dim, n):
        x_input = self.generate_latent_points(latent_dim, n)
        X = generator.predict(x_input).astype('int')
        y = np.zeros((n, 1))

        return X, y

    def generate_real_samples(self, n):
        return np.asarray(random.sample(list(self.pickled_labels), n)), np.ones((n, 1))

    def summarize_performance(self, epoch, generator, discriminator, latent_dim, n=50):
        x_real, y_real = self.generate_real_samples(n)
        _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)

        x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, n)
        _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

        print("Epoch {}: acc_real={:.4f}, acc_fake={:.4f}".format(epoch, acc_real, acc_fake))
        if epoch == 10000:
            print(x_fake)
        elif epoch % 500 == 0:
            print(x_fake[0])

    def train(self, n_epochs=10000, n_batch=128):
        half_batch = int(n_batch / 2)
        print("TRAINING...")
        
        for j in range(1):
            for i in range(1, n_epochs+1):
                x_real, y_real = self.generate_real_samples(half_batch)
                x_fake, y_fake = self.generate_fake_samples(self.g_model, 10, half_batch)

                self.d_model.train_on_batch(x_real, y_real)
                self.d_model.train_on_batch(x_fake, y_fake)

                x_gan = self.generate_latent_points(10, n_batch)
                y_gan = np.ones((n_batch, 1))

                self.gan_model.train_on_batch(x_gan, y_gan)

                if i % 500 == 0:
                    self.summarize_performance(i, self.g_model, self.d_model, 10)
        self.g_model.save('generator_GAN_nois2pos')


if __name__ == "__main__":
    PosGAN = PositionGAN("/nethome/mreich8/VisualEssence/data/generator_data/pkl")
    PosGAN.train()

