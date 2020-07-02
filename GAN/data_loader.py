import os
import json
import sys
import requests
from io import BytesIO
from PIL import Image
import pickle
import numpy as np


class DataLoader:
    def __init__(self, save_directory, n_icons):
        self.save_directory = save_directory
        self.n_icons = n_icons

        if len(sys.argv) > 1:
            self.task_id = int(sys.argv[1])
            self.n_partitions = 2

        self.headers = {
            "Referer": "https://thenounproject.com/search/?q=technology",
            "X-Requested-With": "XMLHttpRequest",
        }

    def download_image_batch(self, page_id, batch_size=100):
        try:
            icons_url = "https://thenounproject.com/featured/icons/?page={}&limit={}".format(page_id, batch_size)
            icons_json = json.loads(requests.get(icons_url, headers=self.headers).text)
            icons = []

            for icon in icons_json['featured_icons']:
                icon_image = (255 - np.asarray(Image.open(BytesIO(requests.get(icon['preview_url']).content)))[:, :, 3])
                for title in icon['tags']:
                    if len(title['slug'].split("-")) == 1:
                        icons.append([icon_image, title['slug']])
                        break
            return icons
        except:
            print("Error in loading page, continuing...")

    def download_dataset(self, info_step=5):
        master_list = []
        pickled_data = open((self.save_directory + '/pkl_dataset_' + str(self.task_id) + '.pkl'), 'wb')

        if len(sys.argv) > 1:
            partition_id = self.task_id*int(self.n_icons / int(self.n_partitions*100))
            c = (self.n_icons / (self.n_partitions*100))-1
        else:
            partition_id = self.n_icons / 100
            c = (self.n_icons / 100) - 1

        for i in range(int(partition_id - c), partition_id+1):
            if i % info_step == 0:
                print("Downloaded {} icons, {}% complete".format(i*100, ((i*100)/self.n_icons)))

            master_list += self.download_image_batch(page_id=i)

        pickle.dump(master_list, pickled_data)

    def collect_dataset(self):
        master_list_all = []
        pickled_data = open((self.save_directory + '/all/pkl_dataset_all' + str(self.task_id) + '.pkl'), 'wb')

        for i in os.listdir(self.save_directory):
            if i.endswith(".pkl"):
                partition_data = pickle.load(open((self.save_directory + '/' + i), 'wb'))
                master_list_all += partition_data

        pickle.dump(master_list_all, pickled_data)

if __name__ == "__main__":
    DataLoader("/nethome/mreich8/VisualEssence/data/gan_data", 500000).download_dataset()


