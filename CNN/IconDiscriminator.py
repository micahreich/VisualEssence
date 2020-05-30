import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import CNN.CNNDataGen

import numpy as np


class IconDiscriminator():
    def preprocess(self, X, y):
        pass

    def construct_model(self, img_height, img_width, n_channels=1, n_classes=2):
        pass

    def train(self, X, y, batch_size=128, epochs=150):
        pass
