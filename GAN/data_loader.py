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

    def download_dataset(self, info_step=500):
        master_list = []
        pickled_data = open((self.save_directory + '/pkl_dataset_' + str(self.task_id) + '.pkl'), 'wb')

        random_ids = random.sample(range(0, 2000000), self.n_icons)
        partition_length = len(random_ids) // self.n_partitions

        for i in range(self.task_id*partition_length, (self.task_id*partition_length) + partition_length):
            if (i+1) % info_step == 0:
                print("Downloaded {} icons, {}% complete".format(i+1, ((i+1) / partition_length)))

            try:
                master_list.extend(self.download_image(random_ids[i]))
            except:
                print("Error, continuing...")

        pickle.dump(master_list, pickled_data)
        pickled_data.close()

    def collect_dataset(self):
        master_list_all = []
        pickled_data = open((self.save_directory + '/all/pkl_dataset_all' + str(self.task_id) + '.pkl'), 'wb')

        for i in os.listdir(self.save_directory):
            if i.endswith(".pkl"):
                partition_data = pickle.load(open((self.save_directory + '/' + i), 'wb'))
                master_list_all += partition_data

        pickle.dump(master_list_all, pickled_data)


if __name__ == "__main__":
    DataLoader("/nethome/mreich8/VisualEssence/data/gan_data", 100).download_image(23)


