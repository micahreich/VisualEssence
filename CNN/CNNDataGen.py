import requests
from PIL import Image
from random import randint
from random import sample
from numpy import asarray


class DatasetGenerator:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def get_images(self, save_directory):

        img_ids = [5]

        for i in range(self.n_samples):
            rand_icon_id = randint(0, 3368879)

            while rand_icon_id in img_ids:
                rand_icon_id = randint(0, 3368879)

            img_ids.append(rand_icon_id)

            img_url = "https://static.thenounproject.com/png/{}-200.png".format(rand_icon_id)

            with open(save_directory + "/" + str(rand_icon_id) + ".png", 'wb') as f:
                f.write(requests.get(img_url).content)
            f.close()

    def create_negative_samples(self, path_to_img_1, path_to_img_2, path_to_img_3):

        # Assuming 200 x 200 images rescaled to 140 x 140 to fit in different locations
        # within the new 200 x 200 random amalgamation
        background = Image.open("blank_icon.png")
        _img_1 = Image.open(path_to_img_1).resize((140, 140), Image.ANTIALIAS)
        _img_2 = Image.open(path_to_img_2).resize((140, 140), Image.ANTIALIAS)
        _img_3 = Image.open(path_to_img_3).resize((140, 140), Image.ANTIALIAS)

        # Assuming 200 x 200 background
        rand_placement_vector = sample(range(0, 60), 2)
        background.paste(_img_1, (rand_placement_vector[0], rand_placement_vector[1]), _img_1)

        rand_placement_vector = sample(range(0, 60), 2)
        background.paste(_img_2, (rand_placement_vector[0], rand_placement_vector[1]), _img_2)

        rand_placement_vector = sample(range(0, 60), 2)
        background.paste(_img_3, (rand_placement_vector[0], rand_placement_vector[1]), _img_3)

        rand_icon_id = randint(0, 3368879)
        background.save("random_icon_" + str(rand_icon_id) + ".png", "PNG")

    def generate_dataset(self, save_directory):
        self.get_images(save_directory)


DG = DatasetGenerator(3)
DG.get_images("cnn_data")
#DG.create_negative_samples("cnn_data/91505.png", "cnn_data/2406058.png", "cnn_data/2730865.png")
