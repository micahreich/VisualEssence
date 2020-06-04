import os
import requests
from skimage.morphology import convex_hull_image
from skimage import io
from skimage.util import invert
from PIL import Image
import numpy as np
from numpy import asarray
import math
import random
from tensorflow.keras.models import load_model


def convex_hull(icon_1_path, icon_2_path, icon_3_path, position_vector):
    icon_1_coords = position_vector[0]  # (x, y) coordinates
    icon_2_coords = position_vector[1]
    icon_3_coords = position_vector[2]

    path_to_blank = "/Users/micahreich/Documents/VisualEssence/CNN/evaluation/blank_icon_2.png"

    # Backgrounds and convex hull for icon 1
    background_1 = Image.open(path_to_blank)  # for convex hull images
    background_1_icon = Image.open(path_to_blank)  # for real icons

    icon_1 = invert(io.imread(icon_1_path, as_gray=True))  # original icon image
    icon_1_chull = asarray(invert(convex_hull_image(icon_1)))  # convex hull icon image
    icon_1_chull_array = asarray(Image.fromarray(icon_1_chull).resize((100, 100)))  # convex hull image as array

    _icon_1 = Image.open(icon_1_path).resize((100, 100), Image.ANTIALIAS)
    background_1.paste(Image.fromarray(icon_1_chull_array), (icon_1_coords[0]-50, icon_1_coords[1]-50))
    background_1_icon.paste(_icon_1, (icon_1_coords[0] - 50, icon_1_coords[1] - 50), _icon_1)

    background_1_array = asarray(background_1)  # background w/ pasted chull image as array
    background_1_icon_array = asarray(background_1_icon)

    # Backgrounds and convex hull for icon 2
    background_2 = Image.open(path_to_blank)  # for convex hull images
    background_2_icon = Image.open(path_to_blank)  # for real icons

    icon_2 = invert(io.imread(icon_2_path, as_gray=True))  # original icon image
    icon_2_chull = asarray(invert(convex_hull_image(icon_2)))  # convex hull icon image
    icon_2_chull_array = asarray(Image.fromarray(icon_2_chull).resize((100, 100)))  # convex hull image as array

    _icon_2 = Image.open(icon_2_path).resize((100, 100), Image.ANTIALIAS)
    background_2.paste(Image.fromarray(icon_2_chull_array), (icon_2_coords[0]-50, icon_2_coords[1]-50))
    background_2_icon.paste(_icon_2, (icon_2_coords[0] - 50, icon_2_coords[1] - 50), _icon_2)

    background_2_array = asarray(background_2)  # background w/ pasted chull image as array
    background_2_icon_array = asarray(background_2_icon)

    # Backgrounds and convex hull for icon 3
    background_3 = Image.open(path_to_blank)  # for convex hull images
    background_3_icon = Image.open(path_to_blank)  # for real icons

    icon_3 = invert(io.imread(icon_3_path, as_gray=True))  # original icon image
    icon_3_chull = asarray(invert(convex_hull_image(icon_3)))  # convex hull icon image
    icon_3_chull_array = asarray(Image.fromarray(icon_3_chull).resize((100, 100)))  # convex hull image as array

    _icon_3 = Image.open(icon_3_path).resize((100, 100), Image.ANTIALIAS)
    background_3.paste(Image.fromarray(icon_3_chull_array), (icon_3_coords[0]-50, icon_3_coords[1]-50))
    background_3_icon.paste(_icon_3, (icon_3_coords[0]-50, icon_3_coords[1]-50), _icon_3)
    background_3_array = asarray(background_3)  # background w/ pasted chull image as array
    background_3_icon_array = asarray(background_3_icon)

    # Create combined image
    overlap_1_chull = np.where(background_1_array == 0, background_1_array, background_2_array)
    overlap_1_icon = np.where(background_1_array == 0, background_1_icon_array, background_2_icon_array)

    overlap_2_icon = np.where(overlap_1_chull == 0, overlap_1_icon, background_3_icon_array)

    f_image = Image.fromarray(overlap_2_icon)
    f_image.save("CH_0.png", "PNG")
    f_image.show()


def get_image(icon_id):
    img_url = "https://static.thenounproject.com/png/{}-200.png".format(icon_id)

    with open(os.path.abspath("P_" + str(icon_id) + ".png"), 'wb') as f:
        f.write(requests.get(img_url).content)
    f.close()

    return os.path.abspath("P_" + str(icon_id) + ".png")


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_random_coords():
    placement_vector = []

    vec_1_x = 100
    vec_1_y = 100
    placement_vector.append([vec_1_x, vec_1_y])

    vec_2_x = random.randint(50, 150)
    vec_2_y = random.randint(50, 150)
    while math.sqrt((vec_2_x - vec_1_x) ** 2) > 100 or math.sqrt((vec_2_x - vec_1_x) ** 2) < 50:
        vec_2_x = random.randint(50, 150)
    while math.sqrt((vec_2_y-vec_1_y)**2) > 25:
        vec_2_y = random.randint(50, 150)

    placement_vector.append([vec_2_x, vec_2_y])

    vec_3_x = random.randint(50, 150)
    vec_3_y = random.randint(50, 150)
    while math.sqrt((vec_3_x - vec_2_x) ** 2) > 100 or math.sqrt((vec_3_x - vec_2_x) ** 2) < 50:
        vec_3_x = random.randint(50, 150)
    while math.sqrt((vec_3_y - vec_2_y) ** 2) > 25:
        vec_3_y = random.randint(50, 150)

    placement_vector.append([vec_3_x, vec_3_y])

    return placement_vector


def make_pred(random, icon_id_1, icon_id_2, icon_id_3):
    if random:
        coords = make_random_coords()
    else:
        coords = [[100, 100],
                  [50, 75],
                  [150, 125]]

    chull_img = convex_hull(
        get_image(icon_id_1),
        get_image(icon_id_2),
        get_image(icon_id_3),
        coords
    )

    os.remove("P_" + str(icon_id_1) + ".png")
    os.remove("P_" + str(icon_id_2) + ".png")
    os.remove("P_" + str(icon_id_3) + ".png")

    """_img_1 = Image.open(os.path.abspath(chull_img))
    _img = _img_1.convert('L')
    _img_arr = np.asarray(_img)

    model = load_model('saved_discriminator')

    return model.predict(_img_arr)"""


if __name__ == "__main__":
    print(
        make_pred(False, 817, 793, 48433)
    )
