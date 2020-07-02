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

        self.pickled_images = np.asarray(
            pickle.load(open((data_directory + '/all/pkl_images_all.pkl'), 'rb'))
        ).reshape((-1, 200, 200, 1))

        self.stacked_images = []
        for triplet in self.pickled_images:
            self.stacked_images.append(
                np.dstack((triplet[0], triplet[1], triplet[2]))
            )

        self.gen_dim = (-1, 1, 200, 200, 3)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0003)

        self.stacked_images = np.asarray(self.stacked_images).reshape((-1, 1, 200, 200, 3))
        self.pickled_labels = np.asarray(self.pickled_labels)

        self.generator = self.define_generator()

        self.discriminator = self.define_discriminator()
        self.discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                   optimizer=self.opt,
                                   metrics=['accuracy'])

        self.gan = self.define_gan()
        self.gan.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                         optimizer=self.opt)

    def define_discriminator(self):
        label_input = tf.keras.layers.Input(shape=6)
        x = tf.keras.layers.Dense(units=512, activation='relu')(label_input)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
        x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

        model = tf.keras.Model(label_input, x)

        return model

    def conv_block(self, input, kernel_size, filters, bn_axis=-1, strides=(2, 2, 2)):
        filter_1, filter_2, filter_3 = filters
        kernel_size_1, kernel_size_2, kernel_size_3 = kernel_size

        x = tf.keras.layers.Conv3D(filter_1, kernel_size_1, padding='same', strides=strides)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_2, kernel_size_2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_3, kernel_size_3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        z = tf.keras.layers.Conv3D(filter_3, kernel_size_3, padding='same', strides=strides)(input)
        z = tf.keras.layers.BatchNormalization()(z)

        x = tf.keras.layers.add([x, z])
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    def identity_block(self, input, kernel_size, filters, bn_axis=-1, strides=(2, 2, 2)):
        filter_1, filter_2, filter_3 = filters
        kernel_size_1, kernel_size_2, kernel_size_3 = kernel_size

        x = tf.keras.layers.Conv3D(filter_1, kernel_size_1, padding='same')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_2, kernel_size_2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv3D(filter_3, kernel_size_3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    def define_generator(self):
        stacked_input = tf.keras.layers.Input(shape=(1, 200, 200, 3))

        # Input Conv Block
        x = tf.keras.layers.ZeroPadding3D(padding=(3, 3, 3))(stacked_input)
        x = tf.keras.layers.Conv3D(64,
                                   (7, 7, 7),
                                   strides=(2, 2, 2),
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1))(x)
        x = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(x)

        # Conv Block 1
        x = self.conv_block(x, [1, 3, 1], [64, 64, 256])

        # Identity x 2
        x = self.identity_block(x, [1, 3, 1], [64, 64, 256])
        x = self.identity_block(x, [1, 3, 1], [64, 64, 256])

        # Conv Block 2
        x = self.conv_block(x, [1, 3, 1], [128, 128, 512])

        # Identity x 3
        x = self.identity_block(x, [1, 3, 1], [128, 128, 512])
        x = self.identity_block(x, [1, 3, 1], [128, 128, 512])
        x = self.identity_block(x, [1, 3, 1], [128, 128, 512])

        # Conv Block 3
        x = self.conv_block(x, [1, 3, 1], [256, 256, 1024])

        # Identity x 5
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])
        x = self.identity_block(x, [1, 3, 1], [256, 256, 1024])

        # Conv Block 4
        x = self.conv_block(x, [1, 3, 1], [512, 512, 2048])

        # Identity x 2
        x = self.identity_block(x, [1, 3, 1], [512, 512, 2048])
        x = self.identity_block(x, [1, 3, 1], [512, 512, 2048])

        # Output
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Dense(units=6, activation='relu')(x)

        model = tf.keras.models.Model(stacked_input, x)

        return model

    def define_gan(self):
        self.discriminator.trainable = False

        gan_input = tf.keras.layers.Input(shape=self.gen_dim)
        gen_label = self.generator(gan_input)
        gan_output = self.discriminator(gen_label)

        gan = tf.keras.Model(gan_input, gan_output)

        return gan

    def display_training_info(self, epoch, num_epochs, d_loss, g_loss):
        print("Epoch {}/{} \n"
              "D loss: {:.4f}, acc: {:.4} \n"
              "G loss: {:.4f}".format(epoch + 1, num_epochs, d_loss[0], 100 * d_loss[1], g_loss))

        random_image_batch = self.stacked_images[np.random.randint(0, self.stacked_images.shape[0], 10)]
        random_image_batch = np.asarray(random_image_batch).reshape((-1, 1, 200, 200, 3))

        print("Sample Generator Positions:\n"
              "{}".format(list(self.generator.predict(random_image_batch))))

    def train(self, epochs=30000, batch_size=128, display_interval=100):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            random_label_batch = self.pickled_labels[np.random.randint(0, self.pickled_labels.shape[0], batch_size)]
            random_label_batch = np.asarray(random_label_batch).reshape((-1, 6))

            random_image_batch = self.stacked_images[np.random.randint(0, self.stacked_images.shape[0], batch_size)]
            random_image_batch = np.asarray(random_image_batch).reshape((-1, 1, 200, 200, 3))

            gen_labels = self.generator.predict(random_image_batch)  # fake images

            #  train discriminator
            d_loss_real = self.discriminator.train_on_batch(random_label_batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_labels, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.gan.train_on_batch(random_image_batch, valid)

            if epoch + 1 % display_interval == 0:
                self.display_training_info(epoch, epochs, d_loss, g_loss)

        self.generator.save('generator_GAN_im2pos')
        print("Model Saved Successfully!")


if __name__ == "__main__":
    PosGAN = PositionGAN("/nethome/mreich8/VisualEssence/data/generator_data/pkl")
    PosGAN.train()
