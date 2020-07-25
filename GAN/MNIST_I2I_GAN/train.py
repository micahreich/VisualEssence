import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import random
from models import I2I_GAN
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


class TrainLib:
    def __init__(self):
        self.epochs = 12000
        self.batch_size = 32
        self.img_size = 28
        self.img_channels = 1
        self.num_classes = 10

        models = I2I_GAN()

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
            losses = {'valid': 'binary_crossentropy',
                      'label': 'binary_crossentropy'}

            self.discriminator = models.build_discriminator()
            self.discriminator.compile(
                loss=losses,
                optimizer=optimizer,
                metrics=['accuracy'])

            self.generator = models.build_generator()

            self.ac_gan = models.build_gan(generator=self.generator, discriminator=self.discriminator)
            self.ac_gan.compile(
                loss=losses,
                optimizer=optimizer)

        print("Loading MNIST dataset...")
        (self.x_train, self.y_train), (_, _) = tf.keras.datasets.mnist.load_data()

        self.x_train = (self.x_train.astype('float32') - 127.5) / 127.5
        self.x_train = np.expand_dims(self.x_train, axis=3)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=self.num_classes)

        print(self.x_train.shape)
        print(self.y_train.shape)

    def generate_latent_noise(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.latent_dim))

    def generate_real_samples(self, batch_size):
        idx = np.random.randint(0, self.x_train.shape[0], batch_size)
        return self.x_train[idx], self.y_train[idx]

    def generate_double_samples(self, batch_size):
        idx = np.random.randint(0, self.x_train.shape[0], 2 * batch_size)
        images, labels = self.x_train[idx], self.y_train[idx]
        image_doubles, label_doubles = np.reshape(images, newshape=(batch_size, 2, 28, 28, 1)), \
                                       np.reshape(labels, newshape=(batch_size, 2, 10))
        return image_doubles, self.double_label_to_single(label_doubles)

    def double_label_to_single(self, y):
        labels = []
        for double in y:
            labels.append(double[0]) if double[0] == double[1] else labels.append(np.add(double[0], double[1]))
        return np.asarray(labels)

    def one_hot_to_int(self, y):
        label = []
        for i in range(len(y)):
            if y[i] == 1:
                label.append(i)
        return label

    def sample_images(self, epoch):
        r, c = 4, 4

        x_double, y_double = self.generate_double_samples(self.batch_size)
        gen_imgs = self.generator.predict([x_double[:, 0], x_double[:, 1]])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title(str(self.one_hot_to_int(y_double[cnt])), fontsize=8)
                cnt += 1
        fig.subplots_adjust(hspace=0.8)
        fig.savefig("images/%d.png" % epoch)
        plt.close()
  
    def train(self, sample_interval=500):
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            #  Train discriminator
            x_real, y_real = self.generate_real_samples(self.batch_size)

            x_double, y_double = self.generate_double_samples(self.batch_size)

            gen_imgs = self.generator.predict([x_double[:, 0], x_double[:, 1]])

            d_loss_real = self.discriminator.train_on_batch(x_real, [valid, y_real])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, y_double])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = self.ac_gan.train_on_batch([x_double[:, 0], x_double[:, 1]], [valid, y_double])

            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0])
                 )

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.generator.save('ac_gan_mnist_md')


if __name__ == "__main__":
    TL = TrainLib()
    TL.train()
