import os
import numpy as np
from PIL import Image


def rmCorrupt(path_to_images):
    badImgs = []
    for i in os.listdir(path_to_images):
        try:
            _img = Image.open(path_to_images + "/" + i)
        except:
            badImgs.append(i)
            print("CORRUPT IMAGE ", i)

    for i in badImgs:
        os.remove(path_to_images + "/" + i)


def rmBlank(path_to_images):
    badImgs = []
    for i in os.listdir(path_to_images):
        img_arr = np.asarray(Image.open(path_to_images + "/" + i))
        if len(np.unique(img_arr)) == 1:
            badImgs.append(i)
            print("BLACK IMAGE ", i)
    for i in badImgs:
        os.remove(path_to_images + "/" + i)


def countType(path_to_images):
    i_count = 0
    r_count = 0

    for i in os.listdir(path_to_images):
        if i[0].upper() == "R":
            r_count += 1
        elif i[0].upper() == "I":
            i_count += 1
    print("POS SAMPLE COUNT ", i_count)
    print("NEG SAMPLE COUNT ", r_count)

countType("/nethome/mreich8/VisualEssence/data/cnn_data_backup/cnn_data")
