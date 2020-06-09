import numpy as np


class PositionRegression():
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def distance(self, op_1, op_2):
        return ((op_1[0]-op_2[0])**2) + ((op_1[1]-op_2[1])**2)

    def euclidian_loss(self, y, y_pred):
        # y, y_pred = [[x, y]
        #              [x, y]
        #              [x, y]]
        euclidian_distances = np.asarray([
            self.distance(y_pred[0], y[0]),
            self.distance(y_pred[1], y[1]),
            self.distance(y_pred[2], y[2])
        ])
        return np.sum(euclidian_distances)

    def train(self):
        pass


if __name__ == "__main__":
    RG = PositionRegression("test")
