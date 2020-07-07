import os
import json
import sys
import requests
from io import BytesIO
from PIL import Image
import pickle
import numpy as np
from lxml import html
import random
import requests


class DataLoader:
    def __init__(self, save_directory, n_icons):
        self.save_directory = save_directory
        self.n_icons = n_icons
        self.task_id = 0
        self.n_partitions = 1

        if len(sys.argv) > 1:
            self.task_id = int(sys.argv[1])
            self.n_partitions = 20

        self.headers = {
            "Referer": "https://thenounproject.com/search/?q=technology",
            "X-Requested-With": "XMLHttpRequest",
        }

    def get_icon_link_and_tag(self, icon_id):
        page = requests.get("https://thenounproject.com/browse/?i={}".format(icon_id))
        tree = html.fromstring(page.content)
        for meta in tree.xpath('//meta'):
            if meta.get("property") == "og:image":
                return meta.get("content"), tree.xpath('//title/text()')[0].split()[0].lower()

    def image_to_array(self, icon_url):
        return np.reshape(
            255 - np.asarray(Image.open(BytesIO(requests.get(icon_url).content)))[:, :, 3], (-1, 200, 200, 1)
        )

    def download_image(self, icon_id):
        icon_url, icon_name = self.get_icon_link_and_tag(icon_id)
        return self.image_to_array(icon_url), icon_name

    def download_dataset(self, info_step=100):
        try:
            os.mkdir(path=self.save_directory + "/partition_{}".format(self.task_id))
        except:
            print("/partition_{} directory already exists!".format(self.task_id))

        images = []
        labels = []

        random.seed(651)
        random_ids = random.sample(range(0, 2000000), self.n_icons)

        partition_length = len(random_ids) // self.n_partitions

        for i in range(self.task_id*partition_length, (self.task_id*partition_length) + partition_length):
            if (i+1) % info_step == 0:
                print("Downloaded {} icons, {:.2f}% complete".format(i+1, ((i+1) / partition_length)*100))
            try:
                image, label = self.download_image(random_ids[i])
                images.append(np.reshape(np.asarray(image), (200, 200, 1)))
                labels.append(label)
            except:
                pass  # do nothing

        np.save(self.save_directory + "/partition_{}/images.npy".format(self.task_id), images)
        np.save(self.save_directory + "/partition_{}/labels.npy".format(self.task_id), labels)

    def collect_dataset(self):
        partition_data = os.listdir(self.save_directory)
        glove_embeddings = dict(np.load(self.save_directory + '/glove/glove_embeddings.npy'))

        images = []
        labels = []

        for i in range(len(partition_data)):
            print("Grabbing {}...".format(partition_data[i]))
            images_npy = np.load(self.save_directory + '/' + partition_data[i] + '/images.npy')
            labels_npy = np.load(self.save_directory + '/' + partition_data[i] + '/labels.npy')

            for item in range(len(images_npy)):
                images.append(
                    np.reshape(self.norm_image(images_npy[item]), (200, 200, 1))
                )

                labels.append(
                    np.reshape(self.get_word_vec(labels_npy[item], glove_embeddings), (300,))
                )

        print("Image array shape: {}"
              "Label array shape: {}".format(np.asarray(images).shape, np.asarray(labels).shape))

        np.save(self.save_directory + '/full_dataset_images.npy', images)
        np.save(self.save_directory + '/full_dataset_labels.npy', labels)

    def get_word_vec(self, word, word_vec_dict):
        try:
            return word_vec_dict[word.lower()]
        except KeyError:
            return None

    def norm_image(self, image_array):
        return (np.asarray(image_array) - 127.5) / 127.5

    def collect_wordvec(self):
        embeddings_dict = {}
        path_to_glove = self.save_directory + '/glove/glove.6B.300d.txt'

        with open(path_to_glove, 'r') as word_vectors:
            for line in word_vectors:
                values = line.split()
                word = values[0]
                word_vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = word_vector

        np.save(self.save_directory + '/glove/glove_embeddings.npy', embeddings_dict)

    def get_pkl_info(self, path_to_file):
        pass


if __name__ == "__main__":
    DataLoader("/nethome/mreich8/VisualEssence/data/gan_data", 200000).download_image(23)


