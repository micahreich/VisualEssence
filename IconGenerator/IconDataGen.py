import requests
from random import randint
import os


class DatasetGenerator:
    def __init__(self, n_samples, save_directory):
        try:
            os.mkdir(save_directory)
        except FileExistsError:
            print("WARNING: directory " + save_directory + " already exists")

        self.n_samples = n_samples

    def get_images(self, save_directory):
        print("\n STARTING ICON COLLECTION")

        img_ids = []
        icon_id = 0

        for i in range(self.n_samples):
            if (i + 1) % 100 == 0:
                print("Grabbed " + str(i + 1) + " images")

            rand_icon_id = randint(0, 3368879)

            while rand_icon_id in img_ids:
                rand_icon_id = randint(0, 3368879)

            img_ids.append(rand_icon_id)

            img_url = "https://static.thenounproject.com/png/{}-200.png".format(rand_icon_id)

            with open(os.path.abspath(save_directory + "/I_" + str(icon_id) + ".png"), 'wb') as f:
                f.write(requests.get(img_url).content)
            f.close()

            icon_id += 1
