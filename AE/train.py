import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import random
from models import I2I_AE
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import pickle


class TrainLib:
    def __init__(self):
        self.epochs = 20000
        self.batch_size = 32
        self.img_size = 28
        self.img_channels = 1
        self.num_classes = 10

        models = I2I_AE()

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.encoder = models.build_encoder()
            self.decoder = models.build_decoder()

            self.autoencoder = models.build_autoencoder(encoder_net=self.encoder, decoder_net=self.decoder)
            self.autoencoder.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(0.0002),
                metrics=[tf.keras.metrics.CosineSimilarity()])

        print("Loading MNIST dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        self.images = tf.keras.backend.resize_images(self.x_train.astype('float32'), 32, 23)
        self.labels = self.y_train.astype('int32')

        print("Training Dataset: " + str(self.images.shape))

    def generate_samples(self, batch_size):
        idx = np.random.randint(0, self.images.shape[0], 2 * batch_size)
        single_images = np.reshape(self.images[idx], (batch_size, 2, self.img_size, self.img_size, 1))
        labels = np.reshape(self.labels[idx], (batch_size, 2))

        return np.concatenate((single_images[:, 0], single_images[:, 1]), axis=3), labels

    def sample_images(self, epoch):
        r, c = 4, 4
        stacked_input, composed_labels = self.generate_samples(r * c)
        composed_images = self.encoder.predict(stacked_input)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(composed_images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title(str(composed_labels[cnt]), fontsize=8)
                cnt += 1
        fig.subplots_adjust(hspace=0.8)
        fig.savefig("images/%d.png" % epoch)
        plt.close()
  
    def train(self, sample_interval=200):
        loss_history = []

        for epoch in range(self.epochs):
            stacked_input, _ = self.generate_samples(self.batch_size)
            ae_loss, accuracy = self.autoencoder.train_on_batch(x=stacked_input, y=stacked_input)

            loss_history.append(ae_loss)

            if epoch % 20 == 0:
                print("Epoch {} [loss: {:.4f}, acc: {:.4f}]".format(epoch, ae_loss, 100*accuracy))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        print("Saving Training History...")
        pickle.dump(loss_history, open("loss_hist", "wb"))

        print("Saving Model...")
        self.encoder.save('mnist_ae')


if __name__ == "__main__":
    TL = TrainLib()
    TL.train()
