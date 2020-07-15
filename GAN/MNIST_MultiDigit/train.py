import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
from models import cGAN
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


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
            losses = {
                "valid_output": tf.keras.losses.BinaryCrossentropy(),
                "digit_output": self.negative_cross_entropy,
            }

            loss_weights = {
                "valid_output": 1.0,
                "digit_output": 1.0
            }

            self.generator = models.build_generator()

            self.discriminator = models.build_discriminator()
            self.discriminator.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

            self.cgan = models.build_cgan(generator=self.generator, discriminator=self.discriminator)
            self.cgan.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

        self.x_train = np.reshape(self.x_train, (-1, self.img_size, self.img_size, self.img_channels))
        self.x_train = self.x_train.astype('float32') / 255.0
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=self.n_classes)

    def negative_cross_entropy(self, y_true, y_pred):
        return -1 * K.categorical_crossentropy(target=y_true, output=y_pred)

    def generate_latent_noise(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.latent_dim))

    def generate_multi_hot(self, batch_size):
        pairs = np.random.randint(self.n_classes, size=(batch_size, 2))  # [(5, 2), (1, 9)]
        multi_hot = np.zeros((batch_size, self.n_classes))   # [(0, 0, 0, ...), (0, 0, 0, ...)]

        for i in range(len(multi_hot)):
            multi_hot[i][pairs[i][0]], multi_hot[i][pairs[i][1]] = 1, 1

        return np.asarray(multi_hot)

    def generate_real_samples(self, batch_size):
        idx = np.random.randint(0, self.x_train.shape[0], batch_size)
        return self.x_train[idx], self.y_train[idx]

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = self.generate_latent_noise(r*c)
        sampled_labels_categorical = self.generate_multi_hot(r*c)

        gen_imgs = self.generator.predict([noise, sampled_labels_categorical])

        sampled_labels = []
        for one_hot in sampled_labels_categorical:
            cat_to_digits = []
            for j in range(len(one_hot)):
                if one_hot[j] == 1:
                    cat_to_digits.append(j)
            sampled_labels.append(cat_to_digits)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title("Digits: {}".format(sampled_labels[cnt]))
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("md_images/%d.png" % epoch, bbox_inches='tight', dpi=200)
        plt.close()

    def train(self, sample_interval=250):
        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            #  Train discriminator
            x_real, y_real = self.generate_real_samples(self.batch_size)
            noise = self.generate_latent_noise(self.batch_size)

            gen_imgs = self.generator.predict([noise, y_real])

            d_loss_real, d_loss_digit_pred_real = self.discriminator.train_on_batch(x=x_real,
                                                                                    y={"valid_output": real, "digit_output": y_real})
            d_loss_fake, d_loss_digit_pred_fake = self.discriminator.train_on_batch(x=gen_imgs,
                                                                                    y={"valid_output": fake, "digit_output": y_real})

            d_loss_real_fake = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss_digit_pred = 0.5 * np.add(d_loss_digit_pred_real, d_loss_digit_pred_fake)

            # Train generator
            y_fake = self.generate_multi_hot(self.batch_size)
            cgan_loss_fake, cgan_loss_digit_fake = self.cgan.train_on_batch(x=[noise, y_fake],
                                                                            y={"valid_output": real, "digit_output": y_fake})

            print("E {}  [D real_fake loss: {}, digit_pred loss: {}]\n[G real_fake loss: {}, digit_pred loss: {}]\n".format(
                epoch,
                d_loss_real_fake,
                -1 * d_loss_digit_pred,
                cgan_loss_fake,
                -1 * cgan_loss_digit_fake))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.generator.save('mnist_multidigit_generator')


if __name__ == "__main__":
    TL = TrainLib()
    TL.train()

