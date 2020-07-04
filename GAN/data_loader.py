import os
import json
import sys
import requests
from io import BytesIO
from PIL import Image
import pickle
import dload
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
        return 255 - np.asarray(Image.open(BytesIO(requests.get(icon_url).content)))[:, :, 3]

    def download_image(self, icon_id):
        icon_url, icon_name = self.get_icon_link_and_tag(icon_id)
        return [self.image_to_array(icon_url), icon_name]

    def download_dataset(self, info_step=20):
        master_list = []
        pickled_data = open((self.save_directory + '/pkl_dataset_' + str(self.task_id) + '.pkl'), 'wb')

        random.seed(652)
        random_ids = random.sample(range(0, 2000000), self.n_icons)

        partition_length = len(random_ids) // self.n_partitions

        for i in range(self.task_id*partition_length, (self.task_id*partition_length) + partition_length):
            if (i+1) % info_step == 0:
                print("Downloaded {} icons, {:.2f}% complete".format(i+1, ((i+1) / partition_length)*100))

            try:
                master_list.extend(self.download_image(random_ids[i]))
            except:
                pass  # do nothing

        pickle.dump(master_list, pickled_data)
        pickled_data.close()

    def collect_dataset(self):
        fnames = os.listdir(self.save_directory)
        for i in fnames:
            if not i.endswith(".pkl"):
                fnames.remove(i)

        master_list_all = []

        for i in range(len(fnames)):
            print("Grabbing {}...".format(fnames[i]))

            partition_data = pickle.load(open((self.save_directory + '/' + fnames[i]), 'rb'))
            _reshaped = np.reshape(np.asarray(partition_data), (-1, 2))
            master_list_all.extend(_reshaped)

            if (i+1) % 5 == 0:
                print("Dumping into file {}".format(i+1))

                pickled_data = open((self.save_directory + '/all/pkl_datasetX_' + str(i+1) + '.pkl'), 'wb')
                pickle.dump(master_list_all, pickled_data, pickle.HIGHEST_PROTOCOL)
                master_list_all = []

                pickled_data.close()

    def collect_wordvec(self):
        embeddings_dict = {}
        pickled_wv = open((self.save_directory + '/glove/glove_300d.pkl'), 'wb')
        path_to_glove = self.save_directory + '/glove/glove.6B.300d.txt'

        with open(path_to_glove, 'r') as word_vectors:
            for line in word_vectors:
                values = line.split()
                word = values[0]
                word_vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = word_vector

        print(len(embeddings_dict))
        pickle.dump(embeddings_dict, pickled_wv, pickle.HIGHEST_PROTOCOL)

    def get_word_vec(self, word, word_vec_dict):
        try:
            return word_vec_dict[word.lower()]
        except KeyError:
            print("Could not find word vector for: {}".format(word))
            return None

    def norm_dataset(self):
        pickled_wv = dict(pickle.load(open((self.save_directory + '/glove/glove_300d.pkl'), 'rb')))
        fnames = os.listdir(self.save_directory + '/all')

        normed_dataset = []

        for i in range(len(fnames)):
            print("Grabbing {}...".format(fnames[i]))
            large_partition = pickle.load(open((self.save_directory + '/all/' + fnames[i]), 'rb'))

            for j in range(len(large_partition)):
                if self.get_word_vec(large_partition[j][1], pickled_wv) is not None:
                    normed_dataset.append([
                        np.asarray(large_partition[j][0]), np.asarray(self.get_word_vec(large_partition[j][1], pickled_wv))
                    ])
            normed_large_partition = open((self.save_directory + '/all_norm/pkl_dataset_' + str(i) + '.pkl'), 'wb')
            pickle.dump(normed_dataset, normed_large_partition, pickle.HIGHEST_PROTOCOL)

            normed_large_partition.close()
            normed_dataset = []

    def get_pkl_info(self):
        for i in os.listdir(self.save_directory + '/all'):
            pickled_data = pickle.load(open((self.save_directory + '/all/' + i), 'rb'))
            #print(np.asarray(np.asarray(pickled_data)[:, 0]).dtype)
            #print(np.asarray(np.asarray(pickled_data)[:, 1]).dtype)


if __name__ == "__main__":
    DataLoader("/nethome/mreich8/VisualEssence/data/gan_data", 100).download_image(23)


