import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
from models import cGAN
import matplotlib.pyplot as plt


class TrainLib:
    def __init__(self):
        self.epochs = 20000
        self.batch_size = 32
        self.img_size = 28
        self.img_channels = 1
        self.latent_dim = 100
        self.n_classes = 10

        print("Loading MNIST dataset...")
        # https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
        with np.load('mnist.npz') as data:
            self.x_train = data['x_train']
            self.y_train = data['y_train']

        models = cGAN()

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.generator = models.build_generator()

            self.discriminator = models.build_discriminator()
            self.discriminator.compile(
                loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                metrics=['accuracy']
            )

            self.cgan = models.build_cgan(generator=self.generator, discriminator=self.discriminator)
            self.cgan.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                metrics=['accuracy']
            )

        self.x_train = np.reshape(self.x_train, (-1, self.img_size, self.img_size, self.img_channels))
        self.x_train = self.x_train.astype('float32') / 255.0
        self.y_train = tf.keras.utils.to_categorical(self.y_train)

    def generate_latent_noise(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.latent_dim))

    def generate_real_samples(self, batch_size):
        idx = np.random.randint(0, self.x_train.shape[0], batch_size)
        return self.x_train[idx], self.y_train[idx]

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.uniform(-1.0, 1.0, size=[r * c, self.latent_dim])

        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        sampled_labels_categorical = tf.keras.utils.to_categorical(sampled_labels)

        gen_imgs = self.generator.predict([noise, sampled_labels_categorical])

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch, bbox_inches='tight', dpi=200)
        plt.close()

    def train(self, sample_interval=500):
        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            #  Train discriminator
            x_real, y_real = self.generate_real_samples(self.batch_size)
            noise = self.generate_latent_noise(self.batch_size)

            gen_imgs = self.generator.predict([noise, y_real])

            d_loss_real = self.discriminator.train_on_batch([x_real, y_real], real)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, y_real], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            y_fake = np.eye(self.n_classes)[np.random.choice(self.n_classes, self.batch_size)]
            cgan_loss, acc = self.cgan.train_on_batch([noise, y_fake], real)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], cgan_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.generator.save('mnist_generator')


if __name__ == "__main__":
    TL = TrainLib()
    TL.train()

