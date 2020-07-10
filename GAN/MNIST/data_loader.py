import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import mnist
import urllib.request
import os
import numpy as np


class DataLoader:
    def __init__(self):
        print("Loading MNIST dataset...")
        # https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
        with np.load('mnist.npz') as data:
            self.x_train = data['x_train']
            self.y_train = data['y_train']


if __name__ == "__main__":
    DataLoader()
