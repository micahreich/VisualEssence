import numpy as np
from PIL import Image
import tensorflow as tf


class DataLib:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.img_size= 72
        self.square_size = 30

    def generate_square(self, t_val=255):
        canvas = np.ones(shape=(self.img_size, self.img_size, 3)) * t_val

        x, y = np.random.randint(0, self.img_size - self.square_size, size=2)
        color = np.eye(3)[np.random.choice(3)] * 250

        for r in range(x, x + self.square_size):
            for c in range(y, y + self.square_size):
                canvas[r, c, :] = color

        return canvas

    def generate_samples(self):
        image_dataset = []

        for sample in range(self.n_samples):
            image_dataset.append(self.generate_square().astype('uint8'))

        np.save("data/squares.npy", np.asarray(image_dataset))


if __name__ == "__main__":
    c = call([
        [(0, 0, 50, 50)],
        [(20, 20, 24, 72)],
    ])
    for i in c.numpy():
        Image.fromarray(i.astype("uint8")).show()
